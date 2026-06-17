import math
from dataclasses import dataclass

import numpy as np

from .rules import (
    BOARD_WIDTH,
    N_CARDS,
    bullheads,
    extract_known_cards,
    extract_unseen_cards,
    get_board,
    get_history_matrix,
    get_round,
    get_scores,
    place_card,
    row_score,
    target_row_index,
)


@dataclass(frozen=True)
class FeatureSpec:
    n_cards: int = N_CARDS
    n_players: int = 4
    n_rounds: int = 10
    board_rows: int = 4
    board_width: int = BOARD_WIDTH
    score_scale: float = 40.0
    row_score_scale: float = 20.0


DEFAULT_FEATURE_SPEC = FeatureSpec()

# Medium V2:
# 3 card masks + board slots + row summaries + sorted row summaries + context
# + 4 dynamic action planes.
FEATURE_DIM = 3 * 104 + 4 * 5 * 2 + 4 * 5 + 4 * 3 + 20 + 4 * 104


def legal_mask_from_hand(hand, spec: FeatureSpec = DEFAULT_FEATURE_SPEC) -> np.ndarray:
    mask = np.zeros(spec.n_cards, dtype=np.bool_)
    for card in hand:
        if isinstance(card, int) and 1 <= card <= spec.n_cards:
            mask[card - 1] = True
    return mask


def extract_policy_features(hand, history, player_idx: int, spec: FeatureSpec = DEFAULT_FEATURE_SPEC):
    """Returns (feature_vector, legal_mask) for a policy model.

    The feature vector intentionally avoids constant per-card planes such as
    card value or bullhead count. With a fixed 104-logit output head, those
    constants can be absorbed by output-specific weights and biases. The only
    104-wide action planes here are state-dependent rule/risk quantities.
    """
    board = get_board(history)
    scores = get_scores(history, spec.n_players)
    round_idx = get_round(history)
    hand_cards = sorted(int(card) for card in hand if isinstance(card, int))
    legal_mask = legal_mask_from_hand(hand_cards, spec)

    parts = [
        _mask_planes(hand_cards, history, board, spec),
        _board_slots(board, spec),
        _row_summaries(board, spec),
        _sorted_row_summaries(board, spec),
        _context_features(hand_cards, history, board, scores, round_idx, player_idx, spec),
        _dynamic_action_planes(hand_cards, history, board, spec),
    ]
    features = np.concatenate(parts).astype(np.float32, copy=False)
    return features, legal_mask


def _mask_planes(hand, history, board, spec):
    hand_mask = np.zeros(spec.n_cards, dtype=np.float32)
    played_mask = np.zeros(spec.n_cards, dtype=np.float32)
    board_mask = np.zeros(spec.n_cards, dtype=np.float32)

    for card in hand:
        if 1 <= card <= spec.n_cards:
            hand_mask[card - 1] = 1.0

    for action_row in get_history_matrix(history):
        for card in action_row:
            if 1 <= card <= spec.n_cards:
                played_mask[card - 1] = 1.0

    for row in board:
        for card in row:
            if 1 <= card <= spec.n_cards:
                board_mask[card - 1] = 1.0

    return np.concatenate([hand_mask, played_mask, board_mask])


def _board_slots(board, spec):
    values = []
    for row_idx in range(spec.board_rows):
        row = board[row_idx] if row_idx < len(board) else []
        for slot_idx in range(spec.board_width):
            if slot_idx < len(row):
                values.extend([row[slot_idx] / spec.n_cards, 1.0])
            else:
                values.extend([0.0, 0.0])
    return np.asarray(values, dtype=np.float32)


def _row_summaries(board, spec):
    values = []
    for row_idx in range(spec.board_rows):
        if row_idx >= len(board) or not board[row_idx]:
            values.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            continue
        row = board[row_idx]
        length = len(row)
        score = row_score(row)
        slots_left = max(0, spec.board_width - length)
        danger = 0.0
        if length >= spec.board_width:
            danger = 1.0 + score / spec.row_score_scale
        elif length == spec.board_width - 1:
            danger = 0.6 + 0.5 * score / spec.row_score_scale
        elif length == spec.board_width - 2:
            danger = 0.25 + 0.25 * score / spec.row_score_scale
        values.extend(
            [
                row[-1] / spec.n_cards,
                length / spec.board_width,
                score / spec.row_score_scale,
                slots_left / spec.board_width,
                danger,
            ]
        )
    return np.asarray(values, dtype=np.float32)


def _sorted_row_summaries(board, spec):
    rows = sorted([row for row in board if row], key=lambda row: row[-1])
    values = []
    for row_idx in range(spec.board_rows):
        if row_idx >= len(rows):
            values.extend([0.0, 0.0, 0.0])
            continue
        row = rows[row_idx]
        values.extend([row[-1] / spec.n_cards, len(row) / spec.board_width, row_score(row) / spec.row_score_scale])
    return np.asarray(values, dtype=np.float32)


def _context_features(hand, history, board, scores, round_idx, player_idx, spec):
    my_score = scores[player_idx] if 0 <= player_idx < len(scores) else 0
    opp_scores = [score for idx, score in enumerate(scores) if idx != player_idx]
    while len(opp_scores) < spec.n_players - 1:
        opp_scores.append(0)
    opp_sorted = sorted(opp_scores)[: spec.n_players - 1]
    min_opp = min(opp_scores) if opp_scores else 0
    max_opp = max(opp_scores) if opp_scores else 0
    mean_opp = sum(opp_scores) / len(opp_scores) if opp_scores else 0.0
    better = sum(1 for score in scores if score < my_score)
    same = sum(1 for score in scores if score == my_score)
    rank = (2 * better + same + 1) / 2.0 if scores else 2.5
    unseen = extract_unseen_cards(hand, history, spec.n_cards)
    player_one_hot = [1.0 if idx == player_idx else 0.0 for idx in range(spec.n_players)]
    board_total = sum(row_score(row) for row in board)
    pressure = sum(_row_pressure(row, spec) for row in board)

    values = [
        round_idx / spec.n_rounds,
        max(0, spec.n_rounds - round_idx) / spec.n_rounds,
        len(hand) / spec.n_rounds,
        my_score / spec.score_scale,
        *(score / spec.score_scale for score in opp_sorted),
        (my_score - min_opp) / spec.score_scale,
        (my_score - mean_opp) / spec.score_scale,
        (my_score - max_opp) / spec.score_scale,
        rank / spec.n_players,
        len(unseen) / spec.n_cards,
        *player_one_hot,
        1.0 if my_score < mean_opp else 0.0,
        1.0 if my_score > mean_opp else 0.0,
        board_total / (spec.board_rows * spec.row_score_scale),
        pressure / 10.0,
    ]
    return np.asarray(values, dtype=np.float32)


def _dynamic_action_planes(hand, history, board, spec):
    immediate = np.zeros(spec.n_cards, dtype=np.float32)
    target_len = np.zeros(spec.n_cards, dtype=np.float32)
    gap_to_tail = np.zeros(spec.n_cards, dtype=np.float32)
    take_prob = np.zeros(spec.n_cards, dtype=np.float32)
    if not board:
        return np.concatenate([immediate, target_len, gap_to_tail, take_prob])

    unseen = extract_unseen_cards(hand, history, spec.n_cards)
    unseen_count = max(1, len(unseen))
    min_tail = min(row[-1] for row in board)

    for card in range(1, spec.n_cards + 1):
        _, penalty, target_idx = place_card(board, card)
        target_row = board[target_idx]
        target_tail = target_row[-1]
        immediate[card - 1] = penalty / spec.row_score_scale
        target_len[card - 1] = len(target_row) / spec.board_width
        gap_to_tail[card - 1] = max(0, card - target_tail) / spec.n_cards
        if card < min_tail or len(target_row) >= spec.board_width:
            take_prob[card - 1] = 1.0
        else:
            interval = sum(1 for unseen_card in unseen if target_tail < unseen_card < card)
            p_interval = interval / unseen_count
            slots = max(0, spec.board_width - len(target_row))
            take_prob[card - 1] = _binomial_tail_at_least(3, p_interval, slots)

    return np.concatenate([immediate, target_len, gap_to_tail, take_prob])


def _row_pressure(row, spec):
    if not row:
        return 0.0
    score = row_score(row)
    if len(row) >= spec.board_width:
        return 1.0 + score / spec.row_score_scale
    if len(row) == spec.board_width - 1:
        return 0.6 + 0.5 * score / spec.row_score_scale
    if len(row) == spec.board_width - 2:
        return 0.25 + 0.25 * score / spec.row_score_scale
    return 0.02 * score / spec.row_score_scale


def _binomial_tail_at_least(n, p, k):
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    p = min(1.0, max(0.0, p))
    total = 0.0
    for hits in range(k, n + 1):
        total += math.comb(n, hits) * (p**hits) * ((1.0 - p) ** (n - hits))
    return float(total)


def action_index(card: int) -> int:
    return int(card) - 1


def index_to_card(index: int) -> int:
    return int(index) + 1
