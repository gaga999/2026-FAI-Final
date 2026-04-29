from __future__ import annotations

import random
import time
from typing import List


class BestPlayer2:
    """Non-NN competitive Take 6 player.

    Design:
    - Weighted heuristic evaluation for immediate and short-horizon risk.
    - Lightweight Monte Carlo rollouts with hidden-card inference.
    - Dynamic rollout budget to stay safe under 1-second action timeout.
    """

    def __init__(
        self,
        player_idx: int,
        seed: int = 0,
        rollout_min: int = 64,
        rollout_max: int = 512,
        deadline_s: float = 0.88,
        cand_keep: int = 7,
        w_immediate: float = 3.2,
        w_future: float = 1.7,
        w_pressure: float = 1.3,
        w_opening: float = 0.22,
    ):
        self.player_idx = int(player_idx)
        self.rng = random.Random(int(seed) + 10007 * self.player_idx)

        self.rollout_min = max(8, int(rollout_min))
        self.rollout_max = max(self.rollout_min, int(rollout_max))
        self.deadline_s = max(0.25, float(deadline_s))
        self.cand_keep = max(3, int(cand_keep))

        self.w_immediate = float(w_immediate)
        self.w_future = float(w_future)
        self.w_pressure = float(w_pressure)
        self.w_opening = float(w_opening)

    def action(self, hand, history):
        if not hand:
            return 1

        cards = sorted(int(c) for c in hand)
        if len(cards) == 1:
            return cards[0]

        board = history.get("board", [])
        if not self._valid_board(board):
            return cards[0]

        start = time.perf_counter()

        try:
            # Phase 1: fast heuristic ranking to prune candidates.
            ranked = self._rank_candidates(cards, board, history)
            keep_n = min(len(ranked), self.cand_keep)
            candidates = [c for c, _ in ranked[:keep_n]]

            # Phase 2: Monte Carlo on top candidates with dynamic budget.
            best_card = candidates[0]
            best_value = float("inf")

            rollout_budget = self._dynamic_rollouts(len(cards), history)
            unknown = self._infer_unknown_cards(cards, board, history)

            # Avoid very costly loops when state is near terminal.
            if not unknown:
                return candidates[0]

            n_opp = 3
            counts = {c: 0 for c in candidates}
            sums = {c: 0.0 for c in candidates}

            idx = 0
            while idx < rollout_budget:
                if time.perf_counter() - start > self.deadline_s:
                    break

                for card in candidates:
                    if time.perf_counter() - start > self.deadline_s:
                        break

                    sample = self._sample_opponent_cards(unknown, n_opp)
                    mc_cost = self._rollout_one_round(card, cards, board, sample, history)
                    sums[card] += mc_cost
                    counts[card] += 1

                idx += 1

            for card in candidates:
                if counts[card] == 0:
                    # Hard fallback to deterministic heuristic.
                    value = dict(ranked).get(card, float("inf"))
                else:
                    mean_cost = sums[card] / float(counts[card])
                    det_cost = dict(ranked).get(card, 0.0)
                    # Blend expected simulation cost with deterministic signal.
                    value = 0.78 * mean_cost + 0.22 * det_cost

                if value < best_value:
                    best_value = value
                    best_card = card
                elif value == best_value and card < best_card:
                    best_card = card

            return best_card
        except Exception:
            # TimeoutException from engine is BaseException, so this does not swallow it.
            return cards[0]

    # ---------- Core game primitives ----------

    def _valid_board(self, board) -> bool:
        if not isinstance(board, list) or len(board) != 4:
            return False
        for row in board:
            if not isinstance(row, list) or len(row) == 0:
                return False
            if not isinstance(row[-1], int):
                return False
        return True

    def _bullheads(self, card: int) -> int:
        if card % 55 == 0:
            return 7
        if card % 11 == 0:
            return 5
        if card % 10 == 0:
            return 3
        if card % 5 == 0:
            return 2
        return 1

    def _row_penalty(self, row: List[int]) -> int:
        return sum(self._bullheads(c) for c in row if isinstance(c, int))

    def _choose_row_to_take(self, board: List[List[int]]) -> int:
        # Match engine tie-break: min score, then min row length, then min index.
        best_idx = 0
        best_key = None
        for i, row in enumerate(board):
            key = (self._row_penalty(row), len(row), i)
            if best_key is None or key < best_key:
                best_key = key
                best_idx = i
        return best_idx

    def _place_card(self, board: List[List[int]], card: int) -> int:
        row_ends = [row[-1] for row in board]

        if card < min(row_ends):
            idx = self._choose_row_to_take(board)
            penalty = self._row_penalty(board[idx])
            board[idx] = [card]
            return penalty

        best_idx = None
        best_end = -1
        for i, row in enumerate(board):
            end = row[-1]
            if end < card and end > best_end:
                best_end = end
                best_idx = i

        if best_idx is None:
            idx = self._choose_row_to_take(board)
            penalty = self._row_penalty(board[idx])
            board[idx] = [card]
            return penalty

        if len(board[best_idx]) >= 5:
            penalty = self._row_penalty(board[best_idx])
            board[best_idx] = [card]
            return penalty

        board[best_idx].append(card)
        return 0

    # ---------- Heuristic evaluation ----------

    def _rank_candidates(self, hand: List[int], board: List[List[int]], history) -> List[tuple[int, float]]:
        out: List[tuple[int, float]] = []

        # Conservative/aggressive modulation based on score standing.
        scores = history.get("scores", [0, 0, 0, 0])
        me = 0
        if isinstance(scores, list) and 0 <= self.player_idx < len(scores):
            me = int(scores[self.player_idx])
        avg_opp = 0.0
        if isinstance(scores, list) and len(scores) >= 2:
            opp_vals = [int(s) for i, s in enumerate(scores) if i != self.player_idx]
            if opp_vals:
                avg_opp = sum(opp_vals) / float(len(opp_vals))
        trailing = 1.0 if me > avg_opp + 4.0 else 0.0
        leading = 1.0 if me + 4.0 < avg_opp else 0.0

        row_ends = [row[-1] for row in board]
        min_end = min(row_ends)

        for card in hand:
            sim = [r[:] for r in board]
            immediate_penalty = float(self._place_card(sim, card))

            # Row-pressure and openness from the resulting board.
            pressure = self._board_pressure(sim)
            opening = self._future_opening(card, hand, sim)

            # Low-card forced take risk in current board.
            forced_take_risk = 0.0
            if card < min_end:
                forced_take_risk = float(min(self._row_penalty(r) for r in board))

            # Slightly more variance when trailing, safer when leading.
            immediate_w = self.w_immediate * (0.92 if trailing else 1.08 if leading else 1.0)
            pressure_w = self.w_pressure * (0.86 if trailing else 1.16 if leading else 1.0)

            score = (
                immediate_w * immediate_penalty
                + self.w_future * forced_take_risk
                + pressure_w * pressure
                - self.w_opening * opening
            )

            # Mild tie-break to retain medium flexibility in early turns.
            score += 0.0015 * abs(card - (sum(hand) / float(len(hand))))
            out.append((card, score))

        out.sort(key=lambda x: (x[1], x[0]))
        return out

    def _board_pressure(self, board: List[List[int]]) -> float:
        # Higher when rows are close to overflow and carry many bullheads.
        p = 0.0
        for row in board:
            length = len(row)
            pen = self._row_penalty(row)
            if length >= 5:
                p += 7.5 + 0.15 * pen
            elif length == 4:
                p += 3.0 + 0.10 * pen
            elif length == 3:
                p += 1.0 + 0.06 * pen
            else:
                p += 0.02 * pen
        return p

    def _future_opening(self, played_card: int, hand: List[int], board_after: List[List[int]]) -> float:
        # Measures how many remaining cards can safely attach without immediate take.
        remaining = [c for c in hand if c != played_card]
        if not remaining:
            return 0.0

        ends = [r[-1] for r in board_after]
        if len(ends) != 4:
            return 0.0

        safe = 0.0
        for c in remaining:
            if c < min(ends):
                continue

            best_idx = None
            best_end = -1
            for i, e in enumerate(ends):
                if e < c and e > best_end:
                    best_end = e
                    best_idx = i
            if best_idx is None:
                continue
            if len(board_after[best_idx]) < 5:
                safe += 1.0
            else:
                safe += 0.15

        return safe

    # ---------- Inference + rollout ----------

    def _infer_unknown_cards(self, hand: List[int], board: List[List[int]], history) -> List[int]:
        known = set(hand)

        for row in board:
            for c in row:
                if isinstance(c, int) and 1 <= c <= 104:
                    known.add(c)

        hmat = history.get("history_matrix", [])
        if isinstance(hmat, list):
            for row in hmat:
                if row is None:
                    continue
                for c in row:
                    if isinstance(c, int) and 1 <= c <= 104:
                        known.add(c)

        return [c for c in range(1, 105) if c not in known]

    def _sample_opponent_cards(self, unknown: List[int], n_opp: int) -> List[int]:
        if not unknown:
            return [1] * n_opp
        k = min(n_opp, len(unknown))
        sample = self.rng.sample(unknown, k)
        if k < n_opp:
            sample.extend([min(unknown)] * (n_opp - k))
        return sample

    def _rollout_one_round(
        self,
        my_card: int,
        hand: List[int],
        board: List[List[int]],
        opp_cards: List[int],
        history,
    ) -> float:
        sim = [r[:] for r in board]

        played = [(-1, c) for c in opp_cards]
        played.append((self.player_idx, my_card))
        played.sort(key=lambda x: x[1])

        my_immediate = 0.0
        for pid, c in played:
            delta = float(self._place_card(sim, c))
            if pid == self.player_idx:
                my_immediate += delta

        # Short-horizon risk estimate on resulting board with remaining hand.
        next_hand = [c for c in hand if c != my_card]
        if not next_hand:
            return my_immediate

        # Use best deterministic next move score as risk proxy.
        pseudo_history = {
            "board": sim,
            "scores": history.get("scores", [0, 0, 0, 0]),
            "round": int(history.get("round", 0)) + 1,
            "history_matrix": history.get("history_matrix", []),
            "board_history": history.get("board_history", []),
            "score_history": history.get("score_history", []),
        }
        ranked_next = self._rank_candidates(next_hand, sim, pseudo_history)
        next_risk = ranked_next[0][1] if ranked_next else 0.0

        return my_immediate + 0.38 * next_risk

    def _dynamic_rollouts(self, hand_size: int, history) -> int:
        round_idx = int(history.get("round", 0))

        # Early rounds are more uncertain; late rounds need fewer samples.
        round_factor = 1.15 if round_idx <= 2 else 1.0 if round_idx <= 6 else 0.82
        hand_factor = 1.12 if hand_size >= 8 else 1.0 if hand_size >= 5 else 0.85

        budget = int(self.rollout_min * round_factor * hand_factor)
        return max(self.rollout_min, min(self.rollout_max, budget))
