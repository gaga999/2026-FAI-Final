import itertools
import random
import time

from .heuristic import heuristic_score, heuristic_select, rank_cards
from .rules import extract_unseen_cards, get_board, get_round, get_scores, place_card, resolve_turn, row_score, target_row_index


OPPONENT_TOP_K = 3
BEAM_WIDTH = 20


def _state_history(board: list[list[int]], scores: list[int], round_idx: int) -> dict:
    return {"board": board, "scores": scores, "round": round_idx, "history_matrix": [], "board_history": [], "score_history": []}


def _sample_hidden_hands(
    hand: list[int],
    history: dict,
    rng: random.Random,
    player_idx: int,
    n_players: int = 4,
) -> list[list[int]]:
    remaining_turns = len(hand)
    unseen = extract_unseen_cards(hand, history)
    rng.shuffle(unseen)
    hands = [[] for _ in range(n_players)]
    cursor = 0
    for player in range(n_players):
        if player == player_idx:
            continue
        hands[player] = sorted(unseen[cursor : cursor + remaining_turns])
        cursor += remaining_turns
    return hands


def _leaf_value(board: list[list[int]], scores: list[int], hands: list[list[int]], player_idx: int, round_idx: int) -> float:
    mine = scores[player_idx]
    opponents = [score for idx, score in enumerate(scores) if idx != player_idx]
    mean_opp = sum(opponents) / len(opponents) if opponents else mine
    my_hand = hands[player_idx]
    state = _state_history(board, scores, round_idx)
    if my_hand:
        future = sum(max(0.0, -heuristic_score(card, my_hand, state, player_idx)) for card in my_hand) / len(my_hand)
    else:
        future = 0.0
    exposure = 0.0
    for row in board:
        if len(row) >= 4:
            exposure += 0.3 * row_score(row)
    return -mine - 0.7 * future - 0.5 * max(0.0, mine - mean_opp) - exposure


def _top_actions(hand: list[int], state: dict, player_idx: int, rng: random.Random) -> list[int]:
    if len(hand) <= OPPONENT_TOP_K:
        return list(hand)
    ranked = []
    for card in hand:
        ranked.append((card, heuristic_score(card, hand, state, player_idx) + rng.uniform(-0.05, 0.05)))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return [card for card, _ in ranked[:OPPONENT_TOP_K]]


def _expand_state(state, player_idx: int, rng: random.Random, deadline: float):
    board, scores, hands, round_idx = state
    if time.perf_counter() >= deadline or not hands[player_idx]:
        return []
    hist = _state_history(board, scores, round_idx)
    action_lists = []
    active_players = []
    for idx, hand in enumerate(hands):
        if not hand:
            continue
        active_players.append(idx)
        action_lists.append(_top_actions(hand, hist, idx, rng))

    children = []
    for combo in itertools.product(*action_lists):
        if time.perf_counter() >= deadline:
            break
        actions = [(card, active_players[pos]) for pos, card in enumerate(combo)]
        new_hands = [list(hand) for hand in hands]
        legal = True
        for card, idx in actions:
            if card not in new_hands[idx]:
                legal = False
                break
            new_hands[idx].remove(card)
        if not legal:
            continue
        new_board, new_scores, _ = resolve_turn(board, scores, actions)
        value = _leaf_value(new_board, new_scores, new_hands, player_idx, round_idx + 1)
        children.append((value, (new_board, new_scores, new_hands, round_idx + 1)))
    children.sort(key=lambda item: item[0], reverse=True)
    return [child for _, child in children[:BEAM_WIDTH]]


def _search_root(root_card: int, hand: list[int], history: dict, player_idx: int, rng: random.Random, deadline: float) -> float:
    board = get_board(history)
    scores = get_scores(history, 4)
    round_idx = get_round(history)
    hidden_hands = _sample_hidden_hands(hand, history, rng, player_idx, 4)
    hands = [list(hand) if idx == player_idx else hidden_hands[idx] for idx in range(4)]
    if root_card not in hands[player_idx]:
        return -9999.0

    hist = _state_history(board, scores, round_idx)
    actions = [(root_card, player_idx)]
    hands[player_idx].remove(root_card)
    for idx in range(4):
        if idx == player_idx or not hands[idx]:
            continue
        card = _top_actions(hands[idx], hist, idx, rng)[0]
        hands[idx].remove(card)
        actions.append((card, idx))

    board, scores, _ = resolve_turn(board, scores, actions)
    beam = [(board, scores, hands, round_idx + 1)]
    horizon = min(3, len(hands[player_idx]))
    for _ in range(horizon):
        if time.perf_counter() >= deadline:
            break
        expanded = []
        for state in beam:
            expanded.extend(_expand_state(state, player_idx, rng, deadline))
        if not expanded:
            break
        expanded.sort(key=lambda state: _leaf_value(state[0], state[1], state[2], player_idx, state[3]), reverse=True)
        beam = expanded[:BEAM_WIDTH]
    if not beam:
        return _leaf_value(board, scores, hands, player_idx, round_idx + 1)
    return sum(_leaf_value(state[0], state[1], state[2], player_idx, state[3]) for state in beam) / len(beam)


def beam_search_select(hand: list[int], history: dict, player_idx: int, deadline: float, fallback: int, seed: int = 17) -> int:
    cards = sorted(int(card) for card in hand)
    if not cards:
        return fallback
    rng = random.Random(seed + 7919 * player_idx + 101 * get_round(history) + sum(cards))
    values = {card: [] for card in cards}
    idx = 0
    while time.perf_counter() < deadline:
        card = cards[idx % len(cards)]
        values[card].append(_search_root(card, cards, history, player_idx, rng, deadline))
        idx += 1
    best = fallback if fallback in cards else cards[0]
    best_value = float("-inf")
    for card in cards:
        if values[card]:
            value = sum(values[card]) / len(values[card])
        else:
            value = heuristic_score(card, cards, history, player_idx)
        if value > best_value or (value == best_value and card < best):
            best_value = value
            best = card
    return best
