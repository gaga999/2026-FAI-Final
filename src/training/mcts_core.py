from __future__ import annotations

import copy
import math
import random
import time
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


def infer_n_cards(state: FullGameState) -> int:
    n_cards = 104
    for row in state.board:
        for c in row:
            if isinstance(c, int):
                n_cards = max(n_cards, c)
    for hand in state.hands:
        for c in hand:
            if isinstance(c, int):
                n_cards = max(n_cards, c)
    for row in state.history_matrix:
        for c in row:
            if isinstance(c, int):
                n_cards = max(n_cards, c)
    return n_cards


def determinize_state_for_player(
    state: FullGameState,
    player_idx: int,
    rng: random.Random,
) -> FullGameState:
    """Builds a determinized hidden-information state from one player's view.

    We keep root player's hand fixed, and sample opponent hands from unknown cards
    consistent with board + played history + own hand.
    """
    s = state.clone()
    n_cards = infer_n_cards(s)

    known = set(s.hands[player_idx])
    for row in s.board:
        for c in row:
            if isinstance(c, int):
                known.add(c)
    for row in s.history_matrix:
        for c in row:
            if isinstance(c, int):
                known.add(c)

    unknown = [c for c in range(1, n_cards + 1) if c not in known]
    rng.shuffle(unknown)

    ptr = 0
    for p in range(s.n_players):
        if p == player_idx:
            continue
        k = len(s.hands[p])
        if ptr + k <= len(unknown):
            s.hands[p] = sorted(unknown[ptr : ptr + k])
            ptr += k
        else:
            # Fallback in pathological cases: sample with replacement from unknown
            if len(unknown) == 0:
                s.hands[p] = [min(s.hands[player_idx])] * k
            else:
                s.hands[p] = sorted(rng.choice(unknown) for _ in range(k))
    return s


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
) -> FullGameState:
    """Rolls out from current state using policy priors for all players."""
    s = state.clone()
    while not s.is_terminal():
        actions = [0] * s.n_players
        hist = history_view(s)
        for p_idx in range(s.n_players):
            hand = s.hands[p_idx]
            if len(hand) == 1:
                actions[p_idx] = hand[0]
                continue

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

        s = transition_with_engine(s, actions)

    return s


def bootstrap_value_after_joint_action(
    state: FullGameState,
    player_idx: int,
    joint_actions: List[int],
    policy_value_net,
    codec,
) -> float:
    """Fast one-step estimate of the acting player's return after a joint action.

    This is much cheaper than rolling the whole game to the end inside every sim.
    """
    next_state = transition_with_engine(state, joint_actions)
    if next_state.is_terminal():
        return -float(next_state.scores[player_idx])

    hist = history_view(next_state)
    hand = next_state.hands[player_idx]
    if not hand:
        return -float(next_state.scores[player_idx])

    x = codec.encode_partial(hand, hist, player_idx).unsqueeze(0)
    with torch.no_grad():
        _, value = policy_value_net(x)
    # Lower score is better, so higher network value means better position.
    return -float(next_state.scores[player_idx]) - float(value.item())


def _player_utility_after_joint_action(
    state: FullGameState,
    next_state: FullGameState,
    player_idx: int,
    policy_value_net,
    codec,
    immediate_weight: float,
) -> float:
    """Player utility after one simultaneous action with immediate-reaction shaping.

    Utility is based on next cumulative score, an immediate penalty signal from this
    transition, and value bootstrap from the next partial observation.
    """
    immediate_penalty = float(next_state.scores[player_idx] - state.scores[player_idx])
    util = -float(next_state.scores[player_idx]) - immediate_weight * immediate_penalty

    if next_state.is_terminal():
        return util

    hand = next_state.hands[player_idx]
    if not hand:
        return util

    hist = history_view(next_state)
    x = codec.encode_partial(hand, hist, player_idx).unsqueeze(0)
    with torch.no_grad():
        _, value = policy_value_net(x)
    util += float(value.item())
    return util


def evaluate_joint_action(
    state: FullGameState,
    joint_actions: List[int],
    policy_value_net,
    codec,
    immediate_weight: float = 1.0,
) -> tuple[FullGameState, List[float]]:
    """Evaluates a full simultaneous action profile using immediate+bootstrap utility."""
    next_state = transition_with_engine(state, joint_actions)
    utils = [
        _player_utility_after_joint_action(
            state=state,
            next_state=next_state,
            player_idx=p_idx,
            policy_value_net=policy_value_net,
            codec=codec,
            immediate_weight=immediate_weight,
        )
        for p_idx in range(state.n_players)
    ]
    return next_state, utils


def run_root_puct(
    state: FullGameState,
    player_idx: int,
    policy_value_net,
    codec,
    sims: int,
    c_puct: float,
    rng: random.Random,
    root_dirichlet_alpha: float = 0.0,
    root_dirichlet_eps: float = 0.0,
    deadline_time: float | None = None,
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

    prior_by_action = {a: float(priors[a - 1]) for a in hand}
    if root_dirichlet_eps > 0.0 and root_dirichlet_alpha > 0.0 and len(hand) > 1:
        noise = [rng.gammavariate(root_dirichlet_alpha, 1.0) for _ in hand]
        noise_sum = sum(noise)
        if noise_sum > 0.0:
            noise = [x / noise_sum for x in noise]
            eps = max(0.0, min(1.0, float(root_dirichlet_eps)))
            for i, a in enumerate(hand):
                prior_by_action[a] = (1.0 - eps) * prior_by_action[a] + eps * noise[i]

    N = {a: 0 for a in hand}
    W = {a: 0.0 for a in hand}
    Q = {a: 0.0 for a in hand}
    opponent_topk = 2
    opponent_policy_weight = 0.08
    immediate_weight = 1.0
    root_vs_opp_weight = 0.5

    max_sims = max(1, int(sims))
    for sim_idx in range(max_sims):
        # Keep at least one simulation when requested, then honor the deadline.
        if deadline_time is not None and sim_idx > 0 and time.perf_counter() >= float(deadline_time):
            break

        total_n = sum(N.values())
        best_a = hand[0]
        best_u = -1e18
        for a in hand:
            p = prior_by_action[a]
            u = Q[a] + c_puct * p * math.sqrt(total_n + 1.0) / (1.0 + N[a])
            if u > best_u:
                best_u = u
                best_a = a

        # Build one simulated round action profile.
        sim_state = state.clone()
        joint = [0] * sim_state.n_players
        round_hist = history_view(sim_state)
        candidate_lists: Dict[int, List[tuple[int, float]]] = {}
        fallback_by_player = [0] * sim_state.n_players

        for p_idx in range(sim_state.n_players):
            p_hand = sim_state.hands[p_idx]
            if p_idx == player_idx:
                joint[p_idx] = best_a
                continue
            if len(p_hand) == 1:
                joint[p_idx] = p_hand[0]
                continue

            px = codec.encode_partial(p_hand, round_hist, p_idx).unsqueeze(0)
            with torch.no_grad():
                plogits, _ = policy_value_net(px)
                plogits = plogits[0]

            pmask = codec.legal_action_mask(p_hand)
            pprobs = masked_policy_distribution(plogits, pmask, temperature=1.0)
            candidates = p_hand
            weights = [float(pprobs[c - 1]) for c in candidates]
            sw = sum(weights)
            if sw <= 0.0:
                fallback = min(candidates)
                fallback_by_player[p_idx] = fallback
                candidate_lists[p_idx] = [(fallback, 1.0)]
            else:
                norm = [w / sw for w in weights]
                pairs = list(zip(candidates, norm))
                pairs.sort(key=lambda x: x[1], reverse=True)
                top = pairs[: max(1, min(opponent_topk, len(pairs)))]
                candidate_lists[p_idx] = top
                fallback_by_player[p_idx] = top[0][0]

        # Opponents pick actions using policy priors plus value-based one-step utility.
        for p_idx in range(sim_state.n_players):
            if p_idx == player_idx:
                continue

            p_hand = sim_state.hands[p_idx]
            if len(p_hand) <= 1:
                joint[p_idx] = p_hand[0]
                continue

            candidates = candidate_lists.get(p_idx, [(fallback_by_player[p_idx], 1.0)])
            if len(candidates) == 1:
                joint[p_idx] = candidates[0][0]
                continue

            if deadline_time is not None and time.perf_counter() >= float(deadline_time):
                joint[p_idx] = candidates[0][0]
                continue

            best_card = candidates[0][0]
            best_score = -1e18
            for cand, pprob in candidates:
                test_joint = list(joint)
                test_joint[player_idx] = best_a
                test_joint[p_idx] = cand
                for q_idx in range(sim_state.n_players):
                    if q_idx == player_idx or q_idx == p_idx:
                        continue
                    q_hand = sim_state.hands[q_idx]
                    if len(q_hand) == 1:
                        test_joint[q_idx] = q_hand[0]
                    else:
                        test_joint[q_idx] = fallback_by_player[q_idx]

                _, test_utils = evaluate_joint_action(
                    state=sim_state,
                    joint_actions=test_joint,
                    policy_value_net=policy_value_net,
                    codec=codec,
                    immediate_weight=immediate_weight,
                )
                score = float(test_utils[p_idx]) + opponent_policy_weight * math.log(max(pprob, 1e-9))
                if score > best_score:
                    best_score = score
                    best_card = cand
            joint[p_idx] = best_card

        _, utilities = evaluate_joint_action(
            state=sim_state,
            joint_actions=joint,
            policy_value_net=policy_value_net,
            codec=codec,
            immediate_weight=immediate_weight,
        )
        opp_utils = [utilities[i] for i in range(sim_state.n_players) if i != player_idx]
        mean_opp = sum(opp_utils) / max(1, len(opp_utils))
        value = float(utilities[player_idx]) - root_vs_opp_weight * float(mean_opp)

        N[best_a] += 1
        W[best_a] += value
        Q[best_a] = W[best_a] / float(N[best_a])

    total = float(sum(N.values()))
    if total <= 0.0:
        return {a: 1.0 / len(hand) for a in hand}
    return {a: N[a] / total for a in hand}


def run_determinized_root_puct(
    state: FullGameState,
    player_idx: int,
    policy_value_net,
    codec,
    total_sims: int,
    c_puct: float,
    rng: random.Random,
    determinization_samples: int = 1,
    root_dirichlet_alpha: float = 0.0,
    root_dirichlet_eps: float = 0.0,
    deadline_time: float | None = None,
) -> Dict[int, float]:
    """Run root PUCT over multiple determinizations and aggregate visit fractions.

    This is the training/inference shared search routine for hidden-information play.
    """
    hand = list(state.hands[player_idx])
    if len(hand) == 1:
        return {hand[0]: 1.0}

    k_det = max(1, int(determinization_samples))
    remaining_sims = max(1, int(total_sims))
    agg = {c: 0.0 for c in hand}
    used_dets = 0

    for det_i in range(k_det):
        if deadline_time is not None and time.perf_counter() >= float(deadline_time):
            break

        det_state = determinize_state_for_player(state, player_idx, rng)
        dets_left = k_det - det_i
        sims_this = max(1, remaining_sims // max(1, dets_left))

        visit_dist = run_root_puct(
            state=det_state,
            player_idx=player_idx,
            policy_value_net=policy_value_net,
            codec=codec,
            sims=sims_this,
            c_puct=c_puct,
            rng=rng,
            root_dirichlet_alpha=root_dirichlet_alpha if det_i == 0 else 0.0,
            root_dirichlet_eps=root_dirichlet_eps if det_i == 0 else 0.0,
            deadline_time=deadline_time,
        )
        remaining_sims = max(0, remaining_sims - sims_this)
        used_dets += 1

        for c in hand:
            agg[c] += float(visit_dist.get(c, 0.0))

    if used_dets <= 0:
        return {a: 1.0 / len(hand) for a in hand}

    merged = {c: agg[c] / float(used_dets) for c in hand}
    total = sum(merged.values())
    if total <= 0.0:
        return {a: 1.0 / len(hand) for a in hand}
    return {c: merged[c] / total for c in hand}
