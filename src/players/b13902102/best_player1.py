from __future__ import annotations

import random
import time
from typing import List


class BestPlayer1:
    """Submission-ready player.

    Modes:
    - fast: policy argmax only (most robust for 1-second budget)
    - strong: lightweight root PUCT with determinization-based opponent sampling
    """

    def __init__(
        self,
        player_idx,
        model_path: str | None = None,
        mode: str = "fast",
        puct_sims: int = 48,
        c_puct: float = 1.5,
        deadline_s: float = 0.9,
        seed: int = 0,
    ):
        self.player_idx = int(player_idx)
        self.mode = str(mode)
        self.puct_sims = int(max(4, puct_sims))
        self.c_puct = float(c_puct)
        self.deadline_s = float(deadline_s)
        self.rng = random.Random(seed + 10007 * self.player_idx)

        self.codec = None
        self.net = None
        self._torch = None
        self._masked_policy_distribution = None

        try:
            import torch as _torch  # type: ignore[reportMissingImports]
            from src.training.policy_value_net import (
                PolicyValueNet,
                masked_policy_distribution,
            )
            from src.training.state_codec import StateCodec

            self._torch = _torch
            self._masked_policy_distribution = masked_policy_distribution
            self.codec = StateCodec()
            self.net = PolicyValueNet(input_dim=self.codec.input_dim, hidden_dim=256)
            self.net.eval()

            if model_path:
                payload = _torch.load(model_path, map_location="cpu")
                if isinstance(payload, dict) and "model_state" in payload:
                    self.net.load_state_dict(payload["model_state"], strict=False)
                elif isinstance(payload, dict):
                    self.net.load_state_dict(payload, strict=False)
        except Exception:
            # Keep fully-functional fallback without torch.
            self.codec = None
            self.net = None
            self._torch = None
            self._masked_policy_distribution = None

    def action(self, hand, history):
        # Do not catch BaseException. TimeoutException from engine must propagate.
        if not hand:
            return 1
        hand = sorted(int(c) for c in hand)
        if len(hand) == 1:
            return hand[0]

        start = time.perf_counter()

        try:
            if self.mode == "strong":
                return self._strong_action(hand, history, start)
            return self._fast_action(hand, history)
        except Exception:
            # Safe fallback for normal runtime errors. This does not swallow TimeoutException
            # because TimeoutException inherits BaseException, not Exception.
            return hand[0]

    def _fast_action(self, hand: List[int], history) -> int:
        if self.net is None or self.codec is None:
            return self._heuristic_action(hand, history)

        x = self.codec.encode_partial(hand, history, self.player_idx).unsqueeze(0)
        with self._torch.no_grad():
            logits, _ = self.net(x)
            logits = logits[0]

        mask = self.codec.legal_action_mask(hand)
        probs = self._masked_policy_distribution(logits, mask, temperature=1.0)
        return max(hand, key=lambda c: float(probs[c - 1]))

    def _strong_action(self, hand: List[int], history, start_time: float) -> int:
        board = history.get("board", [])
        if len(board) != 4:
            return self._fast_action(hand, history)

        if self.net is None or self.codec is None:
            return self._heuristic_action(hand, history)

        x = self.codec.encode_partial(hand, history, self.player_idx).unsqueeze(0)
        with self._torch.no_grad():
            root_logits, _ = self.net(x)
            root_logits = root_logits[0]
        root_mask = self.codec.legal_action_mask(hand)
        root_priors = self._masked_policy_distribution(root_logits, root_mask, temperature=1.0)

        N = {a: 0 for a in hand}
        W = {a: 0.0 for a in hand}
        Q = {a: 0.0 for a in hand}

        for _ in range(self.puct_sims):
            if time.perf_counter() - start_time > self.deadline_s:
                break

            total_n = sum(N.values())
            best_a = hand[0]
            best_u = -1e18
            for a in hand:
                p = float(root_priors[a - 1])
                u = Q[a] + self.c_puct * p * ((total_n + 1.0) ** 0.5) / (1.0 + N[a])
                if u > best_u:
                    best_u = u
                    best_a = a

            value = self._one_step_rollout_value(best_a, hand, history)
            N[best_a] += 1
            W[best_a] += value
            Q[best_a] = W[best_a] / float(N[best_a])

        return max(hand, key=lambda c: N[c])

    def _one_step_rollout_value(self, action_card: int, hand: List[int], history) -> float:
        board = [row[:] for row in history.get("board", [])]

        known = set(hand)
        for row in board:
            for c in row:
                if isinstance(c, int):
                    known.add(c)
        for row in history.get("history_matrix", []):
            if row is None:
                continue
            for c in row:
                if isinstance(c, int) and 1 <= c <= 104:
                    known.add(c)

        unknown = [c for c in range(1, 105) if c not in known]
        n_opp = 3
        if len(unknown) >= n_opp:
            opp_cards = self.rng.sample(unknown, n_opp)
        else:
            opp_cards = unknown + [min(hand)] * max(0, n_opp - len(unknown))

        played = [(-1, c) for c in opp_cards]
        played.append((self.player_idx, action_card))
        played.sort(key=lambda x: x[1])

        gained = 0.0
        for pid, c in played:
            delta = self._place_card(board, c)
            if pid == self.player_idx:
                gained += float(delta)

        # Optional value bootstrap on immediate next state with reduced hand.
        next_hand = [c for c in hand if c != action_card]
        next_history = {
            "board": board,
            "scores": history.get("scores", [0, 0, 0, 0]),
            "round": int(history.get("round", 0)) + 1,
            "history_matrix": history.get("history_matrix", []),
            "board_history": history.get("board_history", []),
            "score_history": history.get("score_history", []),
        }
        if next_hand and self.net is not None and self.codec is not None:
            x = self.codec.encode_partial(next_hand, next_history, self.player_idx).unsqueeze(0)
            with self._torch.no_grad():
                _, v = self.net(x)
            gained += -0.5 * float(v.item())

        # Lower is better in this game.
        return -gained

    def _place_card(self, board, card):
        row_ends = [row[-1] for row in board if row]
        if len(row_ends) != 4:
            return 999

        if card < min(row_ends):
            idx = self._choose_row_to_take(board)
            penalty = self._row_penalty(board[idx])
            board[idx] = [card]
            return penalty

        best_idx = None
        best_end = -1
        for i, row in enumerate(board):
            if not row:
                continue
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

    def _choose_row_to_take(self, board):
        best_idx = 0
        best_key = None
        for i, row in enumerate(board):
            key = (self._row_penalty(row), len(row), i)
            if best_key is None or key < best_key:
                best_key = key
                best_idx = i
        return best_idx

    def _row_penalty(self, row):
        return sum(self._bullheads(c) for c in row if isinstance(c, int))

    def _heuristic_action(self, hand: List[int], history) -> int:
        board = history.get("board", [])
        if len(board) != 4:
            return min(hand)

        # Prefer cards that are less likely to force row-taking right now.
        best_card = hand[0]
        best_cost = float("inf")
        for card in hand:
            sim_board = [row[:] for row in board]
            cost = float(self._place_card(sim_board, card))
            # Mild tie-breaker: keep medium cards for flexibility.
            cost += 0.002 * abs(card - (sum(hand) / max(1, len(hand))))
            if cost < best_cost:
                best_cost = cost
                best_card = card
        return best_card

    def _bullheads(self, card):
        if card % 55 == 0:
            return 7
        if card % 11 == 0:
            return 5
        if card % 10 == 0:
            return 3
        if card % 5 == 0:
            return 2
        return 1