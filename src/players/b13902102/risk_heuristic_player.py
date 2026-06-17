try:
    from .heuristic import heuristic_select
except ImportError:
    from heuristic import heuristic_select


class RiskHeuristicPlayer:
    """Fast deterministic risk heuristic player."""

    def __init__(self, player_idx: int, **_):
        self.player_idx = int(player_idx)

    def action(self, hand, history):
        if not hand:
            return 1
        cards = sorted(int(card) for card in hand)
        if len(cards) == 1:
            return cards[0]
        try:
            card = heuristic_select(cards, history, self.player_idx)
            return card if card in cards else cards[0]
        except Exception:
            return cards[0]
