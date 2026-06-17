from typing import Iterable


N_CARDS = 104
BOARD_WIDTH = 5
BOARD_ROWS = 4
BULLHEAD_TABLE = [0] + [
    7 if card % 55 == 0 else 5 if card % 11 == 0 else 3 if card % 10 == 0 else 2 if card % 5 == 0 else 1
    for card in range(1, N_CARDS + 1)
]


def bullheads(card: int) -> int:
    if 1 <= int(card) <= N_CARDS:
        return BULLHEAD_TABLE[int(card)]
    if card % 55 == 0:
        return 7
    if card % 11 == 0:
        return 5
    if card % 10 == 0:
        return 3
    if card % 5 == 0:
        return 2
    return 1


def row_score(row: Iterable[int]) -> int:
    return sum(bullheads(int(card)) for card in row)


def copy_board(board: list[list[int]]) -> list[list[int]]:
    return [list(row) for row in board]


def normalize_board(board) -> list[list[int]]:
    if not isinstance(board, list):
        return []
    out: list[list[int]] = []
    for row in board[:BOARD_ROWS]:
        if isinstance(row, list) and row:
            clean = [int(card) for card in row if isinstance(card, int)]
            if clean:
                out.append(clean)
    return out


def choose_low_card_row(board: list[list[int]]) -> int:
    return min(range(len(board)), key=lambda i: (row_score(board[i]), len(board[i]), i))


def target_row_index(board: list[list[int]], card: int) -> int:
    best_idx = -1
    best_tail = -1
    for idx, row in enumerate(board):
        tail = row[-1]
        if tail < card and tail > best_tail:
            best_tail = tail
            best_idx = idx
    if best_idx != -1:
        return best_idx
    return choose_low_card_row(board)


def place_card(board: list[list[int]], card: int) -> tuple[list[list[int]], int, int]:
    new_board = copy_board(board)
    if not new_board:
        return [[int(card)]], 0, 0

    card = int(card)
    best_idx = -1
    best_tail = -1
    for idx, row in enumerate(new_board):
        tail = row[-1]
        if tail < card and tail > best_tail:
            best_tail = tail
            best_idx = idx

    if best_idx == -1:
        best_idx = choose_low_card_row(new_board)
        penalty = row_score(new_board[best_idx])
        new_board[best_idx] = [card]
        return new_board, penalty, best_idx

    if len(new_board[best_idx]) >= BOARD_WIDTH:
        penalty = row_score(new_board[best_idx])
        new_board[best_idx] = [card]
        return new_board, penalty, best_idx

    new_board[best_idx].append(card)
    return new_board, 0, best_idx


def resolve_turn(
    board: list[list[int]],
    scores: list[int],
    actions: list[tuple[int, int]],
) -> tuple[list[list[int]], list[int], list[int]]:
    new_board = copy_board(board)
    new_scores = list(scores)
    penalties = [0] * len(new_scores)
    for card, player_idx in sorted(actions, key=lambda item: item[0]):
        new_board, penalty, _ = place_card(new_board, int(card))
        if 0 <= player_idx < len(new_scores):
            new_scores[player_idx] += penalty
            penalties[player_idx] += penalty
    return new_board, new_scores, penalties


def get_board(history) -> list[list[int]]:
    if isinstance(history, dict):
        board = normalize_board(history.get("board", []))
        if board:
            return board
    return []


def get_scores(history, n_players: int = 4) -> list[int]:
    scores = history.get("scores", []) if isinstance(history, dict) else []
    if not isinstance(scores, list):
        scores = []
    clean = [int(score) if isinstance(score, (int, float)) else 0 for score in scores[:n_players]]
    return clean + [0] * max(0, n_players - len(clean))


def get_round(history) -> int:
    if isinstance(history, dict) and isinstance(history.get("round"), int):
        return int(history["round"])
    matrix = get_history_matrix(history)
    return len(matrix)


def get_history_matrix(history) -> list[list[int]]:
    matrix = history.get("history_matrix", []) if isinstance(history, dict) else []
    if not isinstance(matrix, list):
        return []
    out: list[list[int]] = []
    for row in matrix:
        if isinstance(row, list):
            out.append([int(card) for card in row if isinstance(card, int)])
    return out


def get_board_history(history) -> list[list[list[int]]]:
    boards = history.get("board_history", []) if isinstance(history, dict) else []
    if not isinstance(boards, list):
        return []
    return [normalize_board(board) for board in boards if normalize_board(board)]


def get_score_history(history) -> list[list[int]]:
    scores = history.get("score_history", []) if isinstance(history, dict) else []
    if not isinstance(scores, list):
        return []
    out: list[list[int]] = []
    for row in scores:
        if isinstance(row, list):
            out.append([int(score) if isinstance(score, (int, float)) else 0 for score in row])
    return out


def extract_known_cards(hand: list[int], history: dict, n_cards: int = N_CARDS) -> set[int]:
    known = {int(card) for card in hand if isinstance(card, int) and 1 <= card <= n_cards}
    for row in get_board(history):
        known.update(card for card in row if 1 <= card <= n_cards)
    for action_row in get_history_matrix(history):
        known.update(card for card in action_row if 1 <= card <= n_cards)
    for board in get_board_history(history):
        for row in board:
            known.update(card for card in row if 1 <= card <= n_cards)
    return known


def extract_unseen_cards(hand: list[int], history: dict, n_cards: int = N_CARDS) -> list[int]:
    known = extract_known_cards(hand, history, n_cards)
    return [card for card in range(1, n_cards + 1) if card not in known]
