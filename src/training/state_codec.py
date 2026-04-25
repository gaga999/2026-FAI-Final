from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class CodecConfig:
    n_cards: int = 104
    n_players: int = 4
    n_rounds: int = 10
    board_rows: int = 4
    board_capacity: int = 5


class StateCodec:
    """Encodes partial observations for policy/value inference.

    The policy network only sees information available in action(hand, history):
    - own hand
    - board
    - scores
    - round
    - played-card history
    """

    def __init__(self, cfg: CodecConfig | None = None):
        self.cfg = cfg or CodecConfig()

    @property
    def input_dim(self) -> int:
        # hand one-hot + seen one-hot + board last + board length + scores + round
        return (
            self.cfg.n_cards
            + self.cfg.n_cards
            + self.cfg.board_rows
            + self.cfg.board_rows
            + self.cfg.n_players
            + 1
        )

    def legal_action_mask(self, hand: List[int]) -> torch.Tensor:
        mask = torch.zeros(self.cfg.n_cards, dtype=torch.float32)
        for card in hand:
            if 1 <= card <= self.cfg.n_cards:
                mask[card - 1] = 1.0
        return mask

    def encode_partial(
        self,
        hand: List[int],
        history: Dict,
        player_idx: int,
    ) -> torch.Tensor:
        n_cards = self.cfg.n_cards
        n_players = self.cfg.n_players
        board_rows = self.cfg.board_rows

        hand_vec = torch.zeros(n_cards, dtype=torch.float32)
        for card in hand:
            if 1 <= card <= n_cards:
                hand_vec[card - 1] = 1.0

        seen_vec = torch.zeros(n_cards, dtype=torch.float32)
        for card in hand:
            if 1 <= card <= n_cards:
                seen_vec[card - 1] = 1.0

        board = history.get("board", [])
        board_last = torch.zeros(board_rows, dtype=torch.float32)
        board_len = torch.zeros(board_rows, dtype=torch.float32)
        for i in range(min(board_rows, len(board))):
            row = board[i]
            if row:
                c = row[-1]
                if 1 <= c <= n_cards:
                    board_last[i] = float(c) / float(n_cards)
                board_len[i] = float(len(row)) / float(self.cfg.board_capacity)
                for rc in row:
                    if 1 <= rc <= n_cards:
                        seen_vec[rc - 1] = 1.0

        history_matrix = history.get("history_matrix", [])
        for row in history_matrix:
            if row is None:
                continue
            for c in row:
                if isinstance(c, int) and 1 <= c <= n_cards:
                    seen_vec[c - 1] = 1.0

        scores_raw = history.get("scores", [0] * n_players)
        scores = [0.0] * n_players
        for i in range(min(n_players, len(scores_raw))):
            scores[i] = float(scores_raw[i])

        self_score = scores[player_idx] if 0 <= player_idx < n_players else 0.0
        score_vec = torch.zeros(n_players, dtype=torch.float32)
        for i in range(n_players):
            score_vec[i] = (scores[i] - self_score) / 20.0

        round_idx = float(history.get("round", 0)) / max(1, self.cfg.n_rounds)
        round_vec = torch.tensor([round_idx], dtype=torch.float32)

        return torch.cat([hand_vec, seen_vec, board_last, board_len, score_vec, round_vec], dim=0)
