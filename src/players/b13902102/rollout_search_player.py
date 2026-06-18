import time

try:
    from .heuristic import heuristic_select
    from .rollout import bounded_rollout_search
except ImportError:
    from heuristic import heuristic_select
    from rollout import bounded_rollout_search


class RolloutSearchPlayer:
    """Bounded rollout search player with heuristic fallback."""

    def __init__(self, player_idx: int, deadline_s: float = 0.85, seed: int = 0, **_):
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
            deadline = time.perf_counter() + self.deadline_s
            card = bounded_rollout_search(cards, history, self.player_idx, deadline, fallback, self.seed)
            return card if card in cards else fallback
        except Exception:
            return fallback if fallback in cards else cards[0]
