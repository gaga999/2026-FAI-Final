import time

try:
    from .beam_search import beam_search_select
    from .heuristic import heuristic_select
except ImportError:
    from beam_search import beam_search_select
    from heuristic import heuristic_select


class BeamExpectimaxPlayer:
    """Shallow beam/expectimax search player."""

    def __init__(self, player_idx: int, deadline_s: float = 0.8, seed: int = 17, **_):
        self.player_idx = int(player_idx)
        self.deadline_s = min(0.90, max(0.10, float(deadline_s)))
        self.seed = int(seed)

    def action(self, hand, history):
        if not hand:
            return 1
        cards = sorted(int(card) for card in hand)
        if len(cards) == 1:
            return cards[0]
        fallback = cards[0]
        try:
            fallback = heuristic_select(cards, history, self.player_idx)
            card = beam_search_select(
                cards,
                history,
                self.player_idx,
                time.perf_counter() + self.deadline_s,
                fallback,
                self.seed,
            )
            return card if card in cards else fallback
        except Exception:
            return fallback if fallback in cards else cards[0]
