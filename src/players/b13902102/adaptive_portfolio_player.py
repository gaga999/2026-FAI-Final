import random
import time
from typing import List


class AdaptivePortfolioPlayer:
    """Model-free one-turn portfolio search."""

    def __init__(
        self,
        player_idx: int,
        seed: int = 23,
        deadline_s: float = 0.82,
        cand_keep: int = 8,
        samples_min: int = 96,
        samples_max: int = 640,
    ):
        self.player_idx = int(player_idx)
        self.rng = random.Random(int(seed) + 7919 * self.player_idx)
        self.deadline_s = max(0.20, min(0.92, float(deadline_s)))
        self.cand_keep = max(4, int(cand_keep))
        self.samples_min = max(16, int(samples_min))
        self.samples_max = max(self.samples_min, int(samples_max))

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
            ranked = self._rank_candidates(cards, board, history, self.player_idx)
            candidates = [c for c, _ in ranked[: min(len(ranked), self.cand_keep)]]
            unknown = self._infer_unknown_cards(cards, board, history)
            if not unknown:
                return candidates[0]

            budget = self._sample_budget(len(cards), history)
            stats = {c: {"n": 0, "sum": 0.0, "worst": 0.0, "bad": 0} for c in candidates}
            deadline = start + self.deadline_s

            idx = 0
            while idx < budget and time.perf_counter() < deadline:
                for card in candidates:
                    if time.perf_counter() >= deadline:
                        break
                    cost = self._sample_one_turn_cost(card, cards, board, unknown, history)
                    stat = stats[card]
                    stat["n"] += 1
                    stat["sum"] += cost
                    if cost > stat["worst"]:
                        stat["worst"] = cost
                    if cost >= 5.0:
                        stat["bad"] += 1
                idx += 1

            det_scores = dict(ranked)
            best_card = candidates[0]
            best_value = float("inf")
            for card in candidates:
                stat = stats[card]
                if stat["n"]:
                    mean_cost = stat["sum"] / stat["n"]
                    bad_rate = stat["bad"] / stat["n"]
                    robust_cost = mean_cost + 0.18 * stat["worst"] + 2.4 * bad_rate
                else:
                    robust_cost = det_scores.get(card, 999.0)

                value = 0.74 * robust_cost + 0.26 * det_scores.get(card, robust_cost)
                if value < best_value or (value == best_value and card < best_card):
                    best_value = value
                    best_card = card
            return best_card
        except Exception:
            return cards[0]

    def _valid_board(self, board) -> bool:
        if not isinstance(board, list) or len(board) != 4:
            return False
        for row in board:
            if not isinstance(row, list) or not row or not isinstance(row[-1], int):
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

    def _row_score(self, row: List[int]) -> int:
        return sum(self._bullheads(c) for c in row if isinstance(c, int))

    def _choose_row_to_take(self, board: List[List[int]]) -> int:
        return min(range(len(board)), key=lambda i: (self._row_score(board[i]), len(board[i]), i))

    def _place_card(self, board: List[List[int]], card: int) -> int:
        best_idx = None
        best_tail = -1
        for i, row in enumerate(board):
            tail = row[-1]
            if tail < card and tail > best_tail:
                best_tail = tail
                best_idx = i

        if best_idx is None:
            idx = self._choose_row_to_take(board)
            penalty = self._row_score(board[idx])
            board[idx] = [card]
            return penalty

        if len(board[best_idx]) >= 5:
            penalty = self._row_score(board[best_idx])
            board[best_idx] = [card]
            return penalty

        board[best_idx].append(card)
        return 0

    def _rank_candidates(self, hand: List[int], board: List[List[int]], history, eval_player_idx=None) -> List[tuple[int, float]]:
        if eval_player_idx is None:
            eval_player_idx = self.player_idx
        scores = history.get("scores", [0, 0, 0, 0])
        my_score = int(scores[eval_player_idx]) if isinstance(scores, list) and eval_player_idx < len(scores) else 0
        opp_scores = [int(s) for i, s in enumerate(scores) if i != eval_player_idx and isinstance(s, (int, float))]
        mean_opp = sum(opp_scores) / len(opp_scores) if opp_scores else my_score
        leading = my_score + 4.0 < mean_opp
        trailing = my_score > mean_opp + 6.0
        round_idx = int(history.get("round", 0)) if isinstance(history, dict) else 0
        min_tail = min(row[-1] for row in board)

        out = []
        for card in hand:
            sim = [r[:] for r in board]
            immediate = float(self._place_card(sim, card))
            low_take = float(min(self._row_score(r) for r in board)) if card < min_tail else 0.0
            pressure = self._board_pressure(sim)
            opening = self._future_opening(card, hand, sim)
            dump_bonus = self._dump_bonus(card, hand, board, round_idx)

            immediate_w = 3.05 * (1.10 if leading else 0.92 if trailing else 1.0)
            pressure_w = 1.20 * (1.18 if leading else 0.88 if trailing else 1.0)
            score = (
                immediate_w * immediate
                + 1.55 * low_take
                + pressure_w * pressure
                + 0.10 * self._bullheads(card)
                - 0.24 * opening
                - dump_bonus
                - (0.0 if trailing else 0.012 * card)
            )
            score += 0.001 * abs(card - (sum(hand) / len(hand)))
            out.append((card, score))
        out.sort(key=lambda item: (item[1], item[0]))
        return out

    def _board_pressure(self, board: List[List[int]]) -> float:
        pressure = 0.0
        for row in board:
            penalty = self._row_score(row)
            if len(row) >= 5:
                pressure += 7.2 + 0.16 * penalty
            elif len(row) == 4:
                pressure += 2.9 + 0.11 * penalty
            elif len(row) == 3:
                pressure += 0.9 + 0.05 * penalty
            else:
                pressure += 0.015 * penalty
        return pressure

    def _future_opening(self, played_card: int, hand: List[int], board_after: List[List[int]]) -> float:
        remaining = [c for c in hand if c != played_card]
        if not remaining:
            return 0.0
        safe = 0.0
        tails = [r[-1] for r in board_after]
        for card in remaining:
            if card < min(tails):
                continue
            idx = self._target_row(board_after, card)
            if idx is not None and len(board_after[idx]) < 5:
                safe += 1.0
            elif idx is not None:
                safe += 0.12
        return safe

    def _dump_bonus(self, card: int, hand: List[int], board: List[List[int]], round_idx: int) -> float:
        if card >= min(row[-1] for row in board):
            return 0.0
        take_idx = self._choose_row_to_take(board)
        cheap_take = self._row_score(board[take_idx])
        high_bull = self._bullheads(card)
        early = 1.0 if round_idx <= 4 else 0.55
        if cheap_take <= 2:
            return early * (0.65 + 0.25 * high_bull)
        if cheap_take <= 4 and high_bull >= 3:
            return early * 0.45
        return 0.0

    def _target_row(self, board: List[List[int]], card: int):
        best_idx = None
        best_tail = -1
        for i, row in enumerate(board):
            tail = row[-1]
            if tail < card and tail > best_tail:
                best_tail = tail
                best_idx = i
        return best_idx

    def _sample_one_turn_cost(self, my_card, hand, board, unknown, history) -> float:
        sim = [r[:] for r in board]
        played = [(my_card, self.player_idx)]
        for opp_idx in range(4):
            if opp_idx == self.player_idx:
                continue
            opp_hand = self._sample_opp_hand(unknown, len(hand))
            opp_card = self._opponent_policy(opp_hand, board, history, opp_idx)
            played.append((opp_card, opp_idx))

        played.sort(key=lambda item: item[0])
        my_penalty = 0.0
        for card, player_idx in played:
            penalty = float(self._place_card(sim, card))
            if player_idx == self.player_idx:
                my_penalty += penalty

        remaining = [c for c in hand if c != my_card]
        if not remaining:
            return my_penalty
        return my_penalty + 0.34 * self._next_hand_risk(remaining, sim, history)

    def _sample_opp_hand(self, unknown: List[int], hand_size: int) -> List[int]:
        if not unknown:
            return [1]
        k = min(max(1, hand_size), len(unknown))
        return self.rng.sample(unknown, k)

    def _opponent_policy(self, opp_hand: List[int], board: List[List[int]], history, opp_idx: int) -> int:
        if len(opp_hand) == 1:
            return opp_hand[0]
        r = self.rng.random()
        if r < 0.58:
            pseudo = {"board": board, "scores": history.get("scores", [0, 0, 0, 0]), "round": history.get("round", 0)}
            return self._rank_candidates(sorted(opp_hand), board, pseudo, opp_idx)[0][0]
        if r < 0.80:
            return min(opp_hand, key=lambda c: self._immediate_penalty(c, board))
        if r < 0.93:
            return max(opp_hand)
        return self.rng.choice(opp_hand)

    def _immediate_penalty(self, card: int, board: List[List[int]]) -> int:
        sim = [r[:] for r in board]
        return self._place_card(sim, card)

    def _next_hand_risk(self, remaining: List[int], board: List[List[int]], history) -> float:
        pseudo = {
            "board": board,
            "scores": history.get("scores", [0, 0, 0, 0]),
            "round": int(history.get("round", 0)) + 1,
        }
        ranked = self._rank_candidates(remaining, board, pseudo, self.player_idx)
        best_next = ranked[0][1] if ranked else 0.0
        bull_tail = sum(self._bullheads(c) for c in remaining) * 0.07
        low_count = sum(1 for c in remaining if c < min(row[-1] for row in board))
        return max(0.0, best_next) + bull_tail + 0.45 * low_count

    def _infer_unknown_cards(self, hand: List[int], board: List[List[int]], history) -> List[int]:
        known = set(hand)
        for row in board:
            for card in row:
                if isinstance(card, int) and 1 <= card <= 104:
                    known.add(card)
        hmat = history.get("history_matrix", [])
        if isinstance(hmat, list):
            for row in hmat:
                if isinstance(row, list):
                    for card in row:
                        if isinstance(card, int) and 1 <= card <= 104:
                            known.add(card)
        bhist = history.get("board_history", [])
        if isinstance(bhist, list):
            for old_board in bhist:
                if isinstance(old_board, list):
                    for row in old_board:
                        if isinstance(row, list):
                            for card in row:
                                if isinstance(card, int) and 1 <= card <= 104:
                                    known.add(card)
        return [card for card in range(1, 105) if card not in known]

    def _sample_budget(self, hand_size: int, history) -> int:
        round_idx = int(history.get("round", 0)) if isinstance(history, dict) else 0
        round_factor = 1.18 if round_idx <= 2 else 1.0 if round_idx <= 6 else 0.78
        hand_factor = 1.12 if hand_size >= 8 else 1.0 if hand_size >= 5 else 0.82
        budget = int(self.samples_min * round_factor * hand_factor)
        return max(self.samples_min, min(self.samples_max, budget))
