from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Dict, List

import torch

from src.engine import Engine
from .policy_value_net import masked_policy_distribution


def card_penalty(card: int) -> int:
    if card % 55 == 0:
        return 7
    if card % 11 == 0:
        return 5
    if card % 10 == 0:
        return 3
    if card % 5 == 0:
        return 2
    return 1


def row_penalty(row: List[int]) -> int:
    return sum(card_penalty(c) for c in row)


@dataclass
class FullGameState:
    n_players: int
    n_rounds: int
    board_size_x: int
    round_idx: int
    board: List[List[int]]
    scores: List[int]
    hands: List[List[int]]
    history_matrix: List[List[int]]
    board_history: List[List[List[int]]]
    score_history: List[List[int]]

    def clone(self) -> "FullGameState":
        return copy.deepcopy(self)

    def is_terminal(self) -> bool:
        return self.round_idx >= self.n_rounds


class _PlannedActionPlayer:
    def __init__(self):
        self.next_action = None

    def set_action(self, card: int) -> None:
        self.next_action = int(card)

    def action(self, hand, history):
        if self.next_action in hand:
            return int(self.next_action)
        return min(hand)


def state_from_engine(engine) -> FullGameState:
    return FullGameState(
        n_players=engine.n_players,
        n_rounds=engine.n_rounds,
        board_size_x=engine.board_size_x,
        round_idx=engine.round,
        board=[row.copy() for row in engine.board],
        scores=list(engine.scores),
        hands=[h.copy() for h in engine.hands],
        history_matrix=[r.copy() for r in engine.history_matrix],
        board_history=[[r.copy() for r in board] for board in engine.board_history],
        score_history=[s.copy() for s in engine.score_history],
    )


def _engine_from_state(state: FullGameState) -> tuple[Engine, List[_PlannedActionPlayer]]:
    players = [_PlannedActionPlayer() for _ in range(state.n_players)]
    max_seen = 104
    for row in state.board:
        for c in row:
            if isinstance(c, int):
                max_seen = max(max_seen, c)
    for hand in state.hands:
        for c in hand:
            if isinstance(c, int):
                max_seen = max(max_seen, c)
    cfg = {
        "n_cards": max_seen,
        "n_players": state.n_players,
        "n_rounds": state.n_rounds,
        "board_size_x": state.board_size_x,
        "board_size_y": len(state.board),
        "verbose": False,
        "seed": None,
    }
    engine = Engine(cfg, players)
    engine.round = int(state.round_idx)
    engine.scores = list(state.scores)
    engine.board = [row.copy() for row in state.board]
    engine.hands = [h.copy() for h in state.hands]
    engine.history_matrix = [r.copy() for r in state.history_matrix]
    engine.board_history = [[r.copy() for r in b] for b in state.board_history]
    engine.score_history = [s.copy() for s in state.score_history]
    return engine, players


def transition_with_engine(state: FullGameState, joint_actions: List[int]) -> FullGameState:
    if len(joint_actions) != state.n_players:
        raise ValueError("joint_actions length mismatch")

    engine, players = _engine_from_state(state)
    for p_idx, card in enumerate(joint_actions):
        players[p_idx].set_action(card)

    engine.play_round()
    engine.round += 1
    return state_from_engine(engine)


def history_view(state: FullGameState) -> Dict:
    return {
        "board": [row.copy() for row in state.board],
        "scores": list(state.scores),
        "round": state.round_idx,
        "history_matrix": [r.copy() for r in state.history_matrix],
        "board_history": [[r.copy() for r in b] for b in state.board_history],
        "score_history": [s.copy() for s in state.score_history],
    }


def _place_card(state: FullGameState, card: int, player_idx: int) -> int:
    best_row_idx = -1
    max_val_under_card = -1

    for r_idx, row in enumerate(state.board):
        last_card = row[-1]
        if last_card < card and last_card > max_val_under_card:
            max_val_under_card = last_card
            best_row_idx = r_idx

    score_incurred = 0

    if best_row_idx != -1:
        if len(state.board[best_row_idx]) >= state.board_size_x:
            score_incurred = row_penalty(state.board[best_row_idx])
            state.board[best_row_idx] = [card]
        else:
            state.board[best_row_idx].append(card)
    else:
        chosen = min(
            range(len(state.board)),
            key=lambda i: (row_penalty(state.board[i]), len(state.board[i]), i),
        )
        score_incurred = row_penalty(state.board[chosen])
        state.board[chosen] = [card]

    state.scores[player_idx] += score_incurred
    return score_incurred


def apply_joint_action(state: FullGameState, joint_actions: List[int]) -> None:
    """Applies one simultaneous round action to state in-place."""
    if len(joint_actions) != state.n_players:
        raise ValueError("joint_actions length mismatch")

    state.board_history.append([row.copy() for row in state.board])

    for p_idx, card in enumerate(joint_actions):
        if card in state.hands[p_idx]:
            state.hands[p_idx].remove(card)
        else:
            # Safe fallback to maintain valid transition.
            fallback = state.hands[p_idx][0]
            state.hands[p_idx].remove(fallback)
            joint_actions[p_idx] = fallback

    state.history_matrix.append(list(joint_actions))

    for card, p_idx in sorted((c, i) for i, c in enumerate(joint_actions)):
        _place_card(state, card, p_idx)

    state.score_history.append(list(state.scores))
    state.round_idx += 1


def rollout_with_policy(
    state: FullGameState,
    policy_value_net,
    codec,
    rng: random.Random,
    temperature: float = 1.0,
    root_player_idx: int = 0,
) -> FullGameState:
    """Rolls out from current state using policy priors for all players."""
    s = state.clone()
    root_player = int(max(0, min(s.n_players - 1, root_player_idx)))
    while not s.is_terminal():
        actions = [0] * s.n_players
        hist = history_view(s)
        for p_idx in range(s.n_players):
            hand = s.hands[p_idx]
            if len(hand) == 1:
                actions[p_idx] = hand[0]
                continue

            if p_idx == root_player:
                x = codec.encode_partial(hand, hist, p_idx).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = policy_value_net(x)
                    logits = logits[0]

                legal_mask = codec.legal_action_mask(hand)
                probs = masked_policy_distribution(logits, legal_mask, temperature=temperature)
                choices = []
                weights = []
                for c in hand:
                    choices.append(c)
                    weights.append(float(probs[c - 1]))

                sw = sum(weights)
                if sw <= 0.0:
                    actions[p_idx] = min(hand)
                else:
                    norm_weights = [w / sw for w in weights]
                    actions[p_idx] = rng.choices(choices, weights=norm_weights, k=1)[0]
            else:
                # Opponent simulation uses hand-only random policy to avoid hidden-hand network leakage.
                actions[p_idx] = rng.choice(hand)

        s = transition_with_engine(s, actions)

    return s


def run_root_puct(
    state: FullGameState,
    player_idx: int,
    policy_value_net,
    codec,
    sims: int,
    c_puct: float,
    rng: random.Random,
) -> Dict[int, float]:
    """Root-only PUCT for one player under full-information training state.

    Returns a distribution over current hand cards (visit fractions).
    """
    hand = state.hands[player_idx]
    if len(hand) == 1:
        return {hand[0]: 1.0}

    hist = history_view(state)
    x = codec.encode_partial(hand, hist, player_idx).unsqueeze(0)
    with torch.no_grad():
        logits, _ = policy_value_net(x)
        logits = logits[0]

    legal_mask = codec.legal_action_mask(hand)
    priors = masked_policy_distribution(logits, legal_mask, temperature=1.0)

    N = {a: 0 for a in hand}
    W = {a: 0.0 for a in hand}
    Q = {a: 0.0 for a in hand}

    for _ in range(max(1, sims)):
        total_n = sum(N.values())
        best_a = hand[0]
        best_u = -1e18
        for a in hand:
            p = float(priors[a - 1])
            u = Q[a] + c_puct * p * math.sqrt(total_n + 1.0) / (1.0 + N[a])
            if u > best_u:
                best_u = u
                best_a = a

        # Build one simulated round action profile.
        sim_state = state.clone()
        joint = [0] * sim_state.n_players
        for p_idx in range(sim_state.n_players):
            p_hand = sim_state.hands[p_idx]
            if p_idx == player_idx:
                joint[p_idx] = best_a
                continue
            if len(p_hand) == 1:
                joint[p_idx] = p_hand[0]
                continue

            # Opponent policies are sampled from legal hand only to keep partial-observation alignment.
            joint[p_idx] = rng.choice(p_hand)

        sim_state = transition_with_engine(sim_state, joint)
        end_state = rollout_with_policy(
            sim_state,
            policy_value_net,
            codec,
            rng=rng,
            temperature=1.0,
            root_player_idx=player_idx,
        )

        # Lower score is better in this game, so value is negative self score.
        value = -float(end_state.scores[player_idx])

        N[best_a] += 1
        W[best_a] += value
        Q[best_a] = W[best_a] / float(N[best_a])

    total = float(sum(N.values()))
    if total <= 0.0:
        return {a: 1.0 / len(hand) for a in hand}
    return {a: N[a] / total for a in hand}
