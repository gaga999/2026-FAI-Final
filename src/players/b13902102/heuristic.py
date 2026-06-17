from .features import estimate_take_probability
from .rules import (
    BOARD_WIDTH,
    N_CARDS,
    bullheads,
    extract_unseen_cards,
    get_board,
    get_round,
    get_scores,
    place_card,
    row_score,
    target_row_index,
)


WEIGHTS = {
    "expected_penalty": 8.0,
    "certain_take_penalty": 12.0,
    "p_take": 4.0,
    "card_bullhead": 0.7,
    "low_card_penalty": 7.0,
    "future_badness": 1.0,
    "high_card_bonus": 0.6,
    "isolation_bonus": 0.4,
    "score_leader_risk": 0.5,
}


def board_pressure(board: list[list[int]]) -> float:
    pressure = 0.0
    for row in board:
        penalty = row_score(row)
        if len(row) >= BOARD_WIDTH:
            pressure += 5.0 + 0.25 * penalty
        elif len(row) == BOARD_WIDTH - 1:
            pressure += 2.2 + 0.18 * penalty
        elif len(row) == BOARD_WIDTH - 2:
            pressure += 0.8 + 0.08 * penalty
    return pressure


def immediate_profile(card: int, board: list[list[int]], unseen: list[int]) -> tuple[float, float, int, int]:
    after_board, penalty, target = place_card(board, card)
    target_before = target_row_index(board, card)
    target_penalty = row_score(board[target_before])
    p_take = estimate_take_probability(card, board, unseen)
    expected = float(penalty)
    if penalty == 0:
        expected += p_take * target_penalty
    expected += 0.08 * board_pressure(after_board)
    return expected, p_take, penalty, target


def future_badness(card: int, hand: list[int], board: list[list[int]]) -> float:
    after_board, _, _ = place_card(board, card)
    remaining = [other for other in hand if other != card]
    if not remaining or not after_board:
        return 0.0
    min_tail = min(row[-1] for row in after_board)
    low_cards = sum(1 for other in remaining if other < min_tail)
    avg_bull = sum(bullheads(other) for other in remaining) / len(remaining)
    dangerous = 0
    for other in remaining:
        target = target_row_index(after_board, other)
        row = after_board[target]
        if len(row) >= BOARD_WIDTH - 1 and row[-1] < other:
            dangerous += 1
    return 0.5 * low_cards + 0.3 * avg_bull + 0.3 * dangerous


def isolation_bonus(card: int, hand: list[int]) -> float:
    gaps = [abs(card - other) for other in hand if other != card]
    if not gaps:
        return 1.0
    return min(min(gaps), 15) / 15.0


def heuristic_score(card: int, hand: list[int], history_or_state, player_idx: int) -> float:
    if isinstance(history_or_state, dict) and "board" in history_or_state:
        board = get_board(history_or_state)
        scores = get_scores(history_or_state)
        round_idx = get_round(history_or_state)
        unseen = extract_unseen_cards(hand, history_or_state)
    else:
        return -float(card)

    if not board:
        return -float(card)

    expected, p_take, immediate, _ = immediate_profile(card, board, unseen)
    min_tail = min(row[-1] for row in board)
    low_penalty = 1.0 if card < min_tail else 0.0
    high_bonus = card / N_CARDS
    iso_bonus = isolation_bonus(card, hand)
    future = future_badness(card, hand, board)

    my_score = scores[player_idx] if 0 <= player_idx < len(scores) else 0
    opp_scores = [score for idx, score in enumerate(scores) if idx != player_idx]
    mean_opp = sum(opp_scores) / len(opp_scores) if opp_scores else my_score
    leader_risk = max(0.0, mean_opp - my_score) / 10.0
    late = round_idx >= 7
    low_card_weight = WEIGHTS["low_card_penalty"] * (0.8 if late else 1.0)

    value = (
        -WEIGHTS["expected_penalty"] * expected
        -WEIGHTS["certain_take_penalty"] * immediate
        -WEIGHTS["p_take"] * p_take
        -WEIGHTS["card_bullhead"] * bullheads(card)
        -low_card_weight * low_penalty
        -WEIGHTS["future_badness"] * future
        +WEIGHTS["high_card_bonus"] * high_bonus
        +WEIGHTS["isolation_bonus"] * iso_bonus
        +WEIGHTS["score_leader_risk"] * leader_risk * expected
    )
    return value


def rank_cards(hand: list[int], history_or_state, player_idx: int) -> list[tuple[int, float]]:
    cards = sorted(int(card) for card in hand)
    ranked = [(card, heuristic_score(card, cards, history_or_state, player_idx)) for card in cards]
    ranked.sort(key=lambda item: (-item[1], tie_break_key(item[0], cards, history_or_state)))
    return ranked


def tie_break_key(card: int, hand: list[int], history_or_state) -> tuple[float, float, int, int]:
    board = get_board(history_or_state) if isinstance(history_or_state, dict) else []
    unseen = extract_unseen_cards(hand, history_or_state) if isinstance(history_or_state, dict) else []
    if not board:
        return (0.0, 0.0, bullheads(card), card)
    _, p_take, immediate, _ = immediate_profile(card, board, unseen)
    return (float(immediate), p_take, bullheads(card), card)


def heuristic_select(hand: list[int], history_or_state, player_idx: int) -> int:
    cards = sorted(int(card) for card in hand)
    if not cards:
        return 1
    ranked = rank_cards(cards, history_or_state, player_idx)
    return ranked[0][0] if ranked else cards[0]
