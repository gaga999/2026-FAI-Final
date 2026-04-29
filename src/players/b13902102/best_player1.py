"""

Design:
- Fast multi-dimensional heuristic ranking to identify top candidates.
- Lightweight Monte Carlo on best candidates with adaptive budget.
- Robust time management and exception handling.
"""

import random
import time
from typing import List


class BestPlayer1:
    def __init__(
        self,
        player_idx: int,
        seed: int = 0,
        rollout_min: int = 64,
        rollout_max: int = 512,
        deadline_s: float = 0.85,
        cand_keep: int = 8,
        w_immediate: float = 2.8,
        w_future: float = 1.5,
        w_pressure: float = 1.1,
        w_opening: float = 0.18,
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
            # Phase 1: Fast multi-dimensional heuristic ranking.
            ranked = self._rank_candidates(cards, board, history)
            keep_n = min(len(ranked), self.cand_keep)
            candidates = [c for c, _ in ranked[:keep_n]]

            # Phase 2: Monte Carlo refinement on top candidates.
            best_card = candidates[0]
            best_value = float("inf")

            rollout_budget = self._dynamic_rollouts(len(cards), history)
            unknown = self._infer_unknown_cards(cards, board, history)

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
                    mc_cost = self._rollout_one_round(card, cards, board, sample)
                    sums[card] += mc_cost
                    counts[card] += 1

                idx += 1

            # Blend MC and heuristic scores.
            for card in candidates:
                if counts[card] == 0:
                    value = dict(ranked).get(card, float("inf"))
                else:
                    mean_mc = sums[card] / float(counts[card])
                    heur_score = dict(ranked).get(card, 0.0)
                    # 75% MC, 25% heuristic for robustness.
                    value = 0.75 * mean_mc + 0.25 * heur_score

                if value < best_value:
                    best_value = value
                    best_card = card
                elif value == best_value and card < best_card:
                    best_card = card

            return best_card
        except Exception:
            return cards[0]

    # ---------- Game primitives ----------

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
        best_idx = 0
        best_key = None
        for i, row in enumerate(board):
            key = (self._row_penalty(row), len(row), i)
            if best_key is None or key < best_key:
                best_key = key
                best_idx = i
        return best_idx

    def _place_card(self, board: List[List[int]], card: int) -> int:
        """Simulates placing a card on the board, returns penalty incurred."""
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

    # ---------- Multi-dimensional heuristic evaluation ----------

    def _rank_candidates(self, hand: List[int], board: List[List[int]], history) -> List[tuple[int, float]]:
        """Fast heuristic ranking of all hand cards. Returns sorted [(card, score), ...]."""
        out: List[tuple[int, float]] = []

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

            # Multi-dimensional signals.
            pressure = self._board_pressure(sim)
            opening = self._future_opening(card, hand, sim)

            forced_take_risk = 0.0
            if card < min_end:
                forced_take_risk = float(min(self._row_penalty(r) for r in board))

            # Adapt weights based on game state.
            immediate_w = self.w_immediate * (0.90 if trailing else 1.10 if leading else 1.0)
            pressure_w = self.w_pressure * (0.85 if trailing else 1.15 if leading else 1.0)

            score = (
                immediate_w * immediate_penalty
                + self.w_future * forced_take_risk
                + pressure_w * pressure
                - self.w_opening * opening
            )

            # Tie-break for medium-range flexibility.
            score += 0.0012 * abs(card - (sum(hand) / float(len(hand))))
            out.append((card, score))

        out.sort(key=lambda x: (x[1], x[0]))
        return out

    def _board_pressure(self, board: List[List[int]]) -> float:
        """Measures risk from rows close to overflow with high penalties."""
        p = 0.0
        for row in board:
            length = len(row)
            pen = self._row_penalty(row)
            if length >= 5:
                p += 7.0 + 0.14 * pen
            elif length == 4:
                p += 2.8 + 0.09 * pen
            elif length == 3:
                p += 0.9 + 0.05 * pen
            else:
                p += 0.015 * pen
        return p

    def _future_opening(self, played_card: int, hand: List[int], board_after: List[List[int]]) -> float:
        """Counts how many remaining cards can attach without forced take."""
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
                safe += 0.12

        return safe

    # ---------- Monte Carlo + inference ----------

    def _infer_unknown_cards(self, hand: List[int], board: List[List[int]], history) -> List[int]:
        """Identifies cards not yet in hand or played."""
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
        """Randomly samples opponent cards from unknown pool."""
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
    ) -> float:
        """Simulates one round and estimates cost/regret. Combines b13902066's future_risk with deterministic next-move signal."""
        sim = [r[:] for r in board]

        played = [(-1, c) for c in opp_cards]
        played.append((self.player_idx, my_card))
        played.sort(key=lambda x: x[1])

        my_immediate = 0.0
        for pid, c in played:
            delta = float(self._place_card(sim, c))
            if pid == self.player_idx:
                my_immediate += delta

        # Estimate future risk on resulting state + remaining hand.
        next_hand = [c for c in hand if c != my_card]
        if not next_hand:
            return my_immediate

        # b13902066-style future risk: cost of minimum card + bullhead penalty of remaining.
        future_risk_score = 0.0
        min_card = min(next_hand)
        if min_card == min(hand):
            future_risk_score += 0.8
        for c in next_hand:
            future_risk_score += self._bullheads(c) * 0.09

        return my_immediate + 0.35 * future_risk_score

    def _dynamic_rollouts(self, hand_size: int, history) -> int:
        """Adaptive MC budget based on game phase and hand size."""
        round_idx = int(history.get("round", 0))

        round_factor = 1.12 if round_idx <= 2 else 1.0 if round_idx <= 6 else 0.80
        hand_factor = 1.10 if hand_size >= 8 else 1.0 if hand_size >= 5 else 0.82

        budget = int(self.rollout_min * round_factor * hand_factor)
        return max(self.rollout_min, min(self.rollout_max, budget))
