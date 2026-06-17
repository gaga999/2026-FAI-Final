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


FEATURE_NAMES = [
    "round_norm",
    "remaining_turns_norm",
    "my_score_norm",
    "score_gap_min_opp_norm",
    "score_gap_mean_opp_norm",
    "tail_0_norm",
    "tail_1_norm",
    "tail_2_norm",
    "tail_3_norm",
    "len_0_norm",
    "len_1_norm",
    "len_2_norm",
    "len_3_norm",
    "row_score_0_norm",
    "row_score_1_norm",
    "row_score_2_norm",
    "row_score_3_norm",
    "unseen_count_norm",
    "card_norm",
    "card_bullheads_norm",
    "is_low_card",
    "immediate_penalty_norm",
    "target_row_norm",
    "target_tail_norm",
    "target_gap_norm",
    "target_len_norm",
    "target_penalty_norm",
    "slots_left_norm",
    "unseen_interval_norm",
    "interval_probability",
    "take_probability",
    "hand_rank_norm",
    "nearest_own_gap_norm",
]


def _pad(values: list[float], n: int, fill: float = 0.0) -> list[float]:
    return values[:n] + [fill] * max(0, n - len(values))


def binom_tail_at_least(n: int, p: float, k: int) -> float:
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    p = min(1.0, max(0.0, p))
    if n == 3:
        probs = [
            (1 - p) ** 3,
            3 * p * (1 - p) ** 2,
            3 * p * p * (1 - p),
            p**3,
        ]
        return sum(probs[k:])
    total = 0.0
    for hits in range(k, n + 1):
        total += _comb(n, hits) * (p**hits) * ((1 - p) ** (n - hits))
    return total


def _comb(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    out = 1
    for i in range(1, k + 1):
        out = out * (n - k + i) // i
    return out


def estimate_take_probability(card: int, board: list[list[int]], unseen: list[int], n_opponents: int = 3) -> float:
    if not board:
        return 0.0
    tails = [row[-1] for row in board]
    if card < min(tails):
        return 1.0
    idx = target_row_index(board, card)
    row = board[idx]
    if len(row) >= BOARD_WIDTH:
        return 1.0
    tail = row[-1]
    interval = sum(1 for unseen_card in unseen if tail < unseen_card < card)
    p_interval = interval / max(1, len(unseen))
    slots = BOARD_WIDTH - len(row)
    return binom_tail_at_least(n_opponents, p_interval, slots)


def extract_action_features(
    hand: list[int],
    history: dict,
    player_idx: int,
) -> tuple[list[int], list[list[float]]]:
    cards = sorted(int(card) for card in hand)
    board = get_board(history)
    if not cards or not board:
        return cards, [[0.0] * len(FEATURE_NAMES) for _ in cards]

    unseen = extract_unseen_cards(cards, history)
    round_idx = get_round(history)
    remaining = max(0, 10 - round_idx)
    scores = get_scores(history, 4)
    my_score = scores[player_idx] if 0 <= player_idx < len(scores) else 0
    opp_scores = [score for idx, score in enumerate(scores) if idx != player_idx]
    min_opp = min(opp_scores) if opp_scores else 0
    mean_opp = sum(opp_scores) / len(opp_scores) if opp_scores else 0.0
    tails = _pad([row[-1] / N_CARDS for row in board], 4)
    lengths = _pad([len(row) / BOARD_WIDTH for row in board], 4)
    penalties = _pad([row_score(row) / 20.0 for row in board], 4)

    base = [
        round_idx / 10.0,
        remaining / 10.0,
        my_score / 40.0,
        (my_score - min_opp) / 40.0,
        (my_score - mean_opp) / 40.0,
        *tails,
        *lengths,
        *penalties,
        len(unseen) / N_CARDS,
    ]

    features: list[list[float]] = []
    min_tail = min(row[-1] for row in board)
    for rank, card in enumerate(cards):
        _, immediate, target = place_card(board, card)
        target_row = board[target]
        target_tail = target_row[-1]
        target_penalty = row_score(target_row)
        slots = max(0, BOARD_WIDTH - len(target_row))
        interval = sum(1 for unseen_card in unseen if target_tail < unseen_card < card)
        interval_prob = interval / max(1, len(unseen))
        p_take = estimate_take_probability(card, board, unseen)
        own_gaps = [abs(card - other) for other in cards if other != card]
        nearest_gap = min(own_gaps) if own_gaps else N_CARDS
        per_card = [
            card / N_CARDS,
            bullheads(card) / 7.0,
            1.0 if card < min_tail else 0.0,
            immediate / 20.0,
            target / 4.0,
            target_tail / N_CARDS,
            max(0, card - target_tail) / N_CARDS,
            len(target_row) / BOARD_WIDTH,
            target_penalty / 20.0,
            slots / BOARD_WIDTH,
            interval / N_CARDS,
            interval_prob,
            p_take,
            rank / max(1, len(cards) - 1),
            min(nearest_gap, 20) / 20.0,
        ]
        features.append(base + per_card)
    return cards, features
