import random
import time

from .heuristic import heuristic_score, heuristic_select, rank_cards
from .rules import (
    N_CARDS,
    extract_unseen_cards,
    get_board,
    get_round,
    get_scores,
    resolve_turn,
)


def _state_history(board: list[list[int]], scores: list[int], round_idx: int) -> dict:
    return {"board": board, "scores": scores, "round": round_idx, "history_matrix": [], "board_history": [], "score_history": []}


def _sample_hidden_hands(
    hand: list[int],
    history: dict,
    rng: random.Random,
    player_idx: int,
    n_players: int = 4,
) -> list[list[int]]:
    round_idx = get_round(history)
    remaining_turns = len(hand)
    unseen = extract_unseen_cards(hand, history, N_CARDS)
    rng.shuffle(unseen)
    hands = [[] for _ in range(n_players)]
    hands[0] = []
    cursor = 0
    for player in range(n_players):
        if player == player_idx:
            continue
        hands[player] = sorted(unseen[cursor : cursor + remaining_turns])
        cursor += remaining_turns
    _ = round_idx
    return hands


def _opponent_action(hand: list[int], state: dict, player_idx: int, rng: random.Random) -> int:
    if len(hand) == 1:
        return hand[0]
    r = rng.random()
    if r < 0.50:
        return heuristic_select(hand, state, player_idx)
    if r < 0.75:
        board = get_board(state)
        best = min(hand, key=lambda card: _immediate_penalty(card, board))
        return best
    if r < 0.90:
        return max(hand)
    return rng.choice(hand)


def _self_future_action(hand: list[int], state: dict, player_idx: int, rng: random.Random) -> int:
    if len(hand) == 1:
        return hand[0]
    if rng.random() < 0.80:
        return heuristic_select(hand, state, player_idx)
    ranked = rank_cards(hand, state, player_idx)
    top = [card for card, _ in ranked[: min(3, len(ranked))]]
    return rng.choice(top)


def _immediate_penalty(card: int, board: list[list[int]]) -> int:
    from .rules import place_card

    _, penalty, _ = place_card(board, card)
    return penalty


def _rank(scores: list[int], player_idx: int) -> float:
    mine = scores[player_idx]
    better = sum(1 for score in scores if score < mine)
    same = sum(1 for score in scores if score == mine)
    return (2 * better + same + 1) / 2.0


def _utility(scores: list[int], player_idx: int) -> float:
    rank = _rank(scores, player_idx)
    mine = scores[player_idx]
    opponents = [score for idx, score in enumerate(scores) if idx != player_idx]
    mean_opp = sum(opponents) / len(opponents) if opponents else mine
    return -10.0 * rank - mine - 0.5 * max(0.0, mine - mean_opp)


def _rollout_once(root_card: int, hand: list[int], history: dict, player_idx: int, rng: random.Random, deadline: float) -> float:
    n_players = 4
    board = get_board(history)
    scores = get_scores(history, n_players)
    round_idx = get_round(history)
    hidden_hands = _sample_hidden_hands(hand, history, rng, player_idx, n_players)
    hands = [list(hand) if idx == player_idx else hidden_hands[idx] for idx in range(n_players)]
    if root_card not in hands[player_idx]:
        return -999.0

    while hands[player_idx]:
        if time.perf_counter() >= deadline:
            break
        state = _state_history(board, scores, round_idx)
        actions: list[tuple[int, int]] = []
        for idx in range(n_players):
            if not hands[idx]:
                continue
            if idx == player_idx and round_idx == get_round(history):
                card = root_card
            elif idx == player_idx:
                card = _self_future_action(hands[idx], state, idx, rng)
            else:
                card = _opponent_action(hands[idx], state, idx, rng)
            if card not in hands[idx]:
                card = min(hands[idx])
            hands[idx].remove(card)
            actions.append((card, idx))
        board, scores, _ = resolve_turn(board, scores, actions)
        round_idx += 1
    return _utility(scores, player_idx)


def bounded_rollout_search(
    hand: list[int],
    history: dict,
    player_idx: int,
    deadline: float,
    fallback: int,
    seed: int = 0,
) -> int:
    cards = sorted(int(card) for card in hand)
    if not cards:
        return fallback
    rng = random.Random(seed + 1009 * player_idx + 37 * get_round(history) + sum(cards))
    heuristic_values = {card: heuristic_score(card, cards, history, player_idx) for card in cards}
    stats = {card: [0, 0.0, 0.0] for card in cards}

    idx = 0
    while time.perf_counter() < deadline:
        card = cards[idx % len(cards)]
        value = _rollout_once(card, cards, history, player_idx, rng, deadline)
        n, total, total_sq = stats[card]
        stats[card] = [n + 1, total + value, total_sq + value * value]
        idx += 1

    best_card = fallback if fallback in cards else cards[0]
    best_value = float("-inf")
    for card in cards:
        n, total, total_sq = stats[card]
        if n:
            mean = total / n
            var = max(0.0, total_sq / n - mean * mean)
            robust = mean - 0.3 * (var**0.5)
        else:
            robust = heuristic_values[card] * 0.1 - 50.0
        if robust > best_value or (robust == best_value and card < best_card):
            best_value = robust
            best_card = card
    return best_card
