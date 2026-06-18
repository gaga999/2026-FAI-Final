from dataclasses import dataclass

from .rules import BULLHEAD_TABLE, BOARD_WIDTH, get_board, get_round, get_scores


N_PLAYERS = 4


@dataclass(slots=True)
class CompactRow:
    tail: int
    count: int
    penalty: int

    def copy(self):
        return CompactRow(self.tail, self.count, self.penalty)


class CompactState:
    __slots__ = ("rows", "scores", "hands", "round_idx")

    def __init__(self, rows, scores, hands, round_idx):
        self.rows = rows
        self.scores = scores
        self.hands = hands
        self.round_idx = int(round_idx)

    @classmethod
    def from_history(cls, hands, history):
        board = get_board(history)
        rows = [
            CompactRow(row[-1], len(row), sum(BULLHEAD_TABLE[card] for card in row))
            for row in board
        ]
        clean_hands = [sorted(int(card) for card in hand) for hand in hands]
        return cls(rows, get_scores(history, len(clean_hands)), clean_hands, get_round(history))

    @classmethod
    def from_public(cls, board, scores, hands, round_idx):
        rows = [
            CompactRow(row[-1], len(row), sum(BULLHEAD_TABLE[card] for card in row))
            for row in board
        ]
        clean_hands = [sorted(int(card) for card in hand) for hand in hands]
        return cls(rows, list(scores), clean_hands, round_idx)

    def copy(self):
        return CompactState(
            [row.copy() for row in self.rows],
            list(self.scores),
            [list(hand) for hand in self.hands],
            self.round_idx,
        )

    def is_terminal(self):
        return not any(self.hands)

    def place_card(self, card, player_idx):
        card = int(card)
        best_idx = -1
        best_tail = -1
        for idx, row in enumerate(self.rows):
            if row.tail < card and row.tail > best_tail:
                best_idx = idx
                best_tail = row.tail

        if best_idx < 0:
            best_idx = min(
                range(len(self.rows)),
                key=lambda idx: (
                    self.rows[idx].penalty,
                    self.rows[idx].count,
                    idx,
                ),
            )
            row = self.rows[best_idx]
            penalty = row.penalty
            row.tail = card
            row.count = 1
            row.penalty = BULLHEAD_TABLE[card]
        else:
            row = self.rows[best_idx]
            if row.count >= BOARD_WIDTH:
                penalty = row.penalty
                row.tail = card
                row.count = 1
                row.penalty = BULLHEAD_TABLE[card]
            else:
                penalty = 0
                row.tail = card
                row.count += 1
                row.penalty += BULLHEAD_TABLE[card]

        self.scores[player_idx] += penalty
        return penalty

    def play_round(self, actions):
        chosen = []
        for player_idx, card in enumerate(actions):
            if card is None or not self.hands[player_idx]:
                continue
            if card not in self.hands[player_idx]:
                card = self.hands[player_idx][0]
            self.hands[player_idx].remove(card)
            chosen.append((int(card), player_idx))

        penalties = [0] * len(self.hands)
        for card, player_idx in sorted(chosen):
            penalties[player_idx] += self.place_card(card, player_idx)
        self.round_idx += 1
        return penalties

    def observation_key(self, player_idx):
        return (
            self.round_idx,
            tuple((row.tail, row.count, row.penalty) for row in self.rows),
            tuple(self.scores),
            tuple(self.hands[player_idx]),
        )

    def utility(self, player_idx):
        mine = self.scores[player_idx]
        opponents = [score for idx, score in enumerate(self.scores) if idx != player_idx]
        if not opponents:
            return -mine / 20.0

        pairwise = 0.0
        for score in opponents:
            if mine < score:
                pairwise += 1.0
            elif mine > score:
                pairwise -= 1.0
        pairwise /= len(opponents)

        mean_opp = sum(opponents) / len(opponents)
        margin = (mean_opp - mine) / (12.0 + abs(mean_opp - mine))
        absolute = mine / (20.0 + mine)
        return 0.75 * pairwise + 0.21 * margin - 0.04 * absolute


def observation_distance(left, right):
    if left == right:
        return 0.0
    distance = 4.0 * abs(left[0] - right[0])
    for row_a, row_b in zip(left[1], right[1]):
        distance += abs(row_a[0] - row_b[0]) / 12.0
        distance += 1.5 * abs(row_a[1] - row_b[1])
        distance += abs(row_a[2] - row_b[2]) / 5.0
    distance += sum(abs(a - b) for a, b in zip(left[2], right[2])) / 8.0
    if left[3] != right[3]:
        distance += 100.0
    return distance


def public_state_for_hand(hand, history, player_idx, n_players=N_PLAYERS):
    hands = [[] for _ in range(n_players)]
    hands[player_idx] = sorted(int(card) for card in hand)
    return CompactState.from_history(hands, history)
