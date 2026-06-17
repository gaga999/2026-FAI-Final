import random
from dataclasses import dataclass

from .rules import (
    N_CARDS,
    extract_unseen_cards,
    get_board,
    get_round,
    get_scores,
    resolve_turn,
)


@dataclass
class LightGameState:
    board: list
    scores: list
    hands: list
    round_idx: int = 0
    history_matrix: list = None
    board_history: list = None
    score_history: list = None

    def __post_init__(self):
        self.history_matrix = [] if self.history_matrix is None else self.history_matrix
        self.board_history = [] if self.board_history is None else self.board_history
        self.score_history = [] if self.score_history is None else self.score_history

    def clone(self):
        return LightGameState(
            board=[row[:] for row in self.board],
            scores=list(self.scores),
            hands=[hand[:] for hand in self.hands],
            round_idx=self.round_idx,
            history_matrix=[row[:] for row in self.history_matrix],
            board_history=[[row[:] for row in board] for board in self.board_history],
            score_history=[row[:] for row in self.score_history],
        )

    def history_for_player(self):
        return {
            "board": [row[:] for row in self.board],
            "scores": list(self.scores),
            "round": self.round_idx,
            "history_matrix": [row[:] for row in self.history_matrix],
            "board_history": [[row[:] for row in board] for board in self.board_history],
            "score_history": [row[:] for row in self.score_history],
        }

    def is_terminal(self):
        return all(len(hand) == 0 for hand in self.hands)

    def step(self, actions):
        self.board_history.append([row[:] for row in self.board])
        ordered_actions = []
        round_cards = [0] * len(self.hands)
        for player_idx, card in actions:
            if self.hands[player_idx]:
                if card not in self.hands[player_idx]:
                    card = min(self.hands[player_idx])
                self.hands[player_idx].remove(card)
                ordered_actions.append((card, player_idx))
                round_cards[player_idx] = card
        self.history_matrix.append(round_cards)
        self.board, self.scores, _ = resolve_turn(self.board, self.scores, ordered_actions)
        self.score_history.append(list(self.scores))
        self.round_idx += 1

    def ranks(self):
        ranks = []
        for score in self.scores:
            better = sum(1 for other in self.scores if other < score)
            same = sum(1 for other in self.scores if other == score)
            ranks.append((2 * better + same + 1) / 2.0)
        return ranks

    def utility(self, player_idx):
        ranks = self.ranks()
        my_score = self.scores[player_idx]
        opponents = [score for idx, score in enumerate(self.scores) if idx != player_idx]
        mean_opp = sum(opponents) / len(opponents) if opponents else my_score
        return (2.5 - ranks[player_idx]) - 0.03 * my_score - 0.02 * max(0.0, my_score - mean_opp)


def deal_light_game(seed=None, n_players=4, n_rounds=10, n_cards=N_CARDS, board_rows=4):
    rng = random.Random(seed)
    deck = list(range(1, n_cards + 1))
    rng.shuffle(deck)
    board = [[deck.pop()] for _ in range(board_rows)]
    hands = []
    for _ in range(n_players):
        hand = [deck.pop() for _ in range(n_rounds)]
        hands.append(sorted(hand))
    return LightGameState(board=board, scores=[0] * n_players, hands=hands)


def hidden_state_from_history(hand, history, player_idx, rng, n_players=4, n_rounds=10):
    board = get_board(history)
    scores = get_scores(history, n_players)
    round_idx = get_round(history)
    remaining = len(hand)
    unseen = extract_unseen_cards(hand, history, N_CARDS)
    rng.shuffle(unseen)
    hands = [[] for _ in range(n_players)]
    hands[player_idx] = sorted(hand)
    cursor = 0
    for idx in range(n_players):
        if idx == player_idx:
            continue
        hands[idx] = sorted(unseen[cursor : cursor + remaining])
        cursor += remaining
    return LightGameState(board=board, scores=scores, hands=hands, round_idx=round_idx)
