from __future__ import annotations

import random
import sys
import time
from typing import List


class BestPlayer1:
    """Submission-ready player with 1-second constrained inference.

    Modes:
    - fast: policy argmax only
    - strong: determinized root PUCT aligned with training search flow
    """

    def __init__(
        self,
        player_idx,
        model_path: str | None = None,
        mode: str = "strong",
        puct_sims: int = 96,
        c_puct: float = 1.5,
        deadline_s: float = 0.88,
        determinization_samples: int = 3,
        seed: int = 0,
    ):
        self.player_idx = int(player_idx)
        self.mode = str(mode)
        self.puct_sims = int(max(4, puct_sims))
        self.c_puct = float(c_puct)
        self.deadline_s = float(deadline_s)
        self.determinization_samples = int(max(1, determinization_samples))
        self.rng = random.Random(seed + 10007 * self.player_idx)

        self.codec = None
        self.net = None
        self._torch = None
        self._masked_policy_distribution = None
        self._FullGameState = None
        self._run_determinized_root_puct = None
        self._init_warning_printed = False
        self._init_warning_msg = None

        try:
            import torch as _torch  # type: ignore[reportMissingImports]
            from src.training.mcts_core import FullGameState, run_determinized_root_puct
            from src.training.policy_value_net import (
                PolicyValueNet,
                masked_policy_distribution,
            )
            from src.training.state_codec import CodecConfig, StateCodec

            self._torch = _torch
            self._masked_policy_distribution = masked_policy_distribution
            self._FullGameState = FullGameState
            self._run_determinized_root_puct = run_determinized_root_puct

            # Keep CPU/RAM usage conservative for submission constraints.
            try:
                _torch.set_num_threads(1)
                if hasattr(_torch, "set_num_interop_threads"):
                    _torch.set_num_interop_threads(1)
            except Exception:
                pass

            cfg = CodecConfig()
            hidden_dim = 256
            state_dict = None

            if model_path:
                payload = _torch.load(model_path, map_location="cpu")
                if isinstance(payload, dict) and "model_state" in payload:
                    codec_meta = payload.get("codec")
                    if not isinstance(codec_meta, dict):
                        raise ValueError("Checkpoint missing codec metadata; retrain with codec v2+")
                    version = int(codec_meta.get("version", -1))
                    if version < 2:
                        raise ValueError(f"Unsupported codec version: {version}")
                    cfg = CodecConfig(
                        n_cards=int(payload.get("n_cards", 104)),
                        n_players=int(codec_meta.get("n_players", 4)),
                        n_rounds=int(codec_meta.get("n_rounds", 10)),
                        board_rows=int(codec_meta.get("board_rows", 4)),
                        board_capacity=int(codec_meta.get("board_capacity", 5)),
                        history_k=int(codec_meta.get("history_k", 4)),
                        codec_version=version,
                    )
                    hidden_dim = int(payload.get("hidden_dim", 256))
                    state_dict = payload["model_state"]
                elif isinstance(payload, dict):
                    raise ValueError("Raw state_dict checkpoint is not supported; expected training payload with codec metadata")

            self.codec = StateCodec(cfg)
            self.net = PolicyValueNet(input_dim=self.codec.input_dim, hidden_dim=hidden_dim, n_cards=cfg.n_cards)
            self.net.eval()
            if state_dict is not None:
                self.net.load_state_dict(state_dict, strict=True)
            if model_path is None:
                self._init_warning_msg = "[BestPlayer1][ERROR] No model_path provided; fallback will use hand[0]."
        except Exception as e:
            # Keep robust action() fallback behavior even without NN.
            self.codec = None
            self.net = None
            self._torch = None
            self._masked_policy_distribution = None
            self._FullGameState = None
            self._run_determinized_root_puct = None
            self._init_warning_msg = f"[BestPlayer1][ERROR] NN disabled, fallback will use hand[0]. reason={type(e).__name__}: {e}"

    def action(self, hand, history):
        if (self._init_warning_msg is not None) and (not self._init_warning_printed):
            print(self._init_warning_msg, file=sys.stderr)
            self._init_warning_printed = True

        # Do not catch BaseException. TimeoutException from engine must propagate.
        if not hand:
            return 1
        hand = sorted(int(c) for c in hand)
        if len(hand) == 1:
            return hand[0]

        if self.net is None or self.codec is None or self._torch is None:
            return self._fallback_hand0(hand, "model-not-loaded")

        start = time.perf_counter()

        try:
            if self.mode == "strong":
                return self._strong_action(hand, history, start)
            return self._fast_action(hand, history)
        except Exception as e:
            # Safe fallback for normal runtime errors. This does not swallow TimeoutException
            # because TimeoutException inherits BaseException, not Exception.
            return self._fallback_hand0(hand, f"runtime-error:{type(e).__name__}:{e}")

    def _fallback_hand0(self, hand: List[int], reason: str) -> int:
        print(f"[BestPlayer1][ERROR] fallback -> hand[0], reason={reason}", file=sys.stderr)
        return hand[0]

    def _fast_action(self, hand: List[int], history) -> int:
        if self.net is None or self.codec is None or self._torch is None:
            return self._fallback_hand0(hand, "fast-mode-net-unavailable")

        x = self.codec.encode_partial(hand, history, self.player_idx).unsqueeze(0)
        with self._torch.no_grad():
            logits, _ = self.net(x)
            logits = logits[0]

        mask = self.codec.legal_action_mask(hand)
        probs = self._masked_policy_distribution(logits, mask, temperature=1.0)
        return max(hand, key=lambda c: float(probs[c - 1]))

    def _strong_action(self, hand: List[int], history, start_time: float) -> int:
        if self.net is None or self.codec is None or self._torch is None:
            return self._fallback_hand0(hand, "strong-mode-net-unavailable")

        if self._run_determinized_root_puct is None or self._FullGameState is None:
            return self._fast_action(hand, history)

        state = self._build_search_state(hand, history)
        if state is None:
            return self._fast_action(hand, history)

        deadline_time = start_time + max(0.05, min(0.98, self.deadline_s))
        visit_dist = self._run_determinized_root_puct(
            state=state,
            player_idx=self.player_idx,
            policy_value_net=self.net,
            codec=self.codec,
            total_sims=self.puct_sims,
            c_puct=self.c_puct,
            rng=self.rng,
            determinization_samples=self.determinization_samples,
            root_dirichlet_alpha=0.0,
            root_dirichlet_eps=0.0,
            deadline_time=deadline_time,
        )

        if not visit_dist:
            return self._fast_action(hand, history)
        best = max(hand, key=lambda c: float(visit_dist.get(c, 0.0)))
        if best not in hand:
            return self._fallback_hand0(hand, "strong-invalid-action")
        return best

    def _build_search_state(self, hand: List[int], history):
        if self.codec is None or self._FullGameState is None:
            return None

        board = history.get("board", [])
        if not isinstance(board, list) or len(board) != int(self.codec.cfg.board_rows):
            return None
        if any((not isinstance(row, list)) or len(row) == 0 for row in board):
            return None

        round_idx = int(history.get("round", 0))
        n_rounds = int(self.codec.cfg.n_rounds)
        remaining_cards = len(hand)
        if round_idx < 0:
            round_idx = 0
        if round_idx > n_rounds:
            round_idx = n_rounds
        if n_rounds - round_idx != remaining_cards:
            # Prefer consistent card count from observed hand for robust inference.
            round_idx = max(0, n_rounds - remaining_cards)

        n_players = int(self.codec.cfg.n_players)
        hands = []
        for p_idx in range(n_players):
            if p_idx == self.player_idx:
                hands.append(list(hand))
            else:
                # Placeholder values; determinization replaces opponent hands by card-count only.
                hands.append([1] * remaining_cards)

        scores_raw = history.get("scores", [0] * n_players)
        scores = [0] * n_players
        for i in range(min(n_players, len(scores_raw))):
            scores[i] = int(scores_raw[i])

        history_matrix_raw = history.get("history_matrix", [])
        history_matrix = []
        if isinstance(history_matrix_raw, list):
            for row in history_matrix_raw:
                if isinstance(row, list):
                    rr = [0] * n_players
                    for i in range(min(n_players, len(row))):
                        rr[i] = int(row[i]) if isinstance(row[i], int) else 0
                    history_matrix.append(rr)

        board_history_raw = history.get("board_history", [])
        board_history = []
        if isinstance(board_history_raw, list):
            for b in board_history_raw:
                if isinstance(b, list):
                    board_history.append([list(r) for r in b if isinstance(r, list)])

        score_history_raw = history.get("score_history", [])
        score_history = []
        if isinstance(score_history_raw, list):
            for s in score_history_raw:
                if isinstance(s, list):
                    rr = [0] * n_players
                    for i in range(min(n_players, len(s))):
                        rr[i] = int(s[i]) if isinstance(s[i], int) else 0
                    score_history.append(rr)

        return self._FullGameState(
            n_players=n_players,
            n_rounds=n_rounds,
            board_size_x=int(self.codec.cfg.board_capacity),
            round_idx=round_idx,
            board=[list(row) for row in board],
            scores=scores,
            hands=hands,
            history_matrix=history_matrix,
            board_history=board_history,
            score_history=score_history,
        )

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
        if card == 55:
            return 55
        if card % 11 == 0:
            return 22
        if card % 10 == 0:
            return 10
        if card % 5 == 0:
            return 5
        return 1