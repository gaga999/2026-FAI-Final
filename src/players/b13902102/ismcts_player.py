import random

try:
    from .heuristic import heuristic_select
    from .ismcts_search import (
        ISMCTSConfig,
        advance_reused_root,
        search_ismcts,
    )
except ImportError:
    from heuristic import heuristic_select
    from ismcts_search import ISMCTSConfig, advance_reused_root, search_ismcts


class ISMCTSPlayer:
    """Model-free belief-weighted ISMCTS player for simultaneous 6 Nimmt turns."""

    def __init__(
        self,
        player_idx,
        seed=13902102,
        deadline_s=0.9,
        max_iterations=20000,
        c_puct=2.25,
        root_warmup_passes=3,
        particle_count=128,
        proposal_count=192,
        tree_horizon=5,
        **kwargs,
    ):
        self.player_idx = int(player_idx)
        self.rng = random.Random(int(seed) + 104729 * self.player_idx)
        self.config = ISMCTSConfig(
            deadline_s=deadline_s,
            max_iterations=max_iterations,
            c_puct=c_puct,
            root_warmup_passes=root_warmup_passes,
            particle_count=particle_count,
            proposal_count=proposal_count,
            tree_horizon=tree_horizon,
            **kwargs,
        )
        self._previous_root = None
        self._previous_card = None
        self.last_search_stats = None

    def action(self, hand, history):
        if not hand:
            return 1
        cards = sorted(int(card) for card in hand)
        if len(cards) == 1:
            self._previous_root = None
            self._previous_card = cards[0]
            return cards[0]

        fallback = cards[0]
        try:
            fallback = heuristic_select(cards, history, self.player_idx)
            reused = advance_reused_root(
                self._previous_root,
                self._previous_card,
                cards,
                history,
                self.player_idx,
            )
            selected, root, diagnostics = search_ismcts(
                cards,
                history,
                self.player_idx,
                self.config,
                self.rng,
                reused_root=reused,
            )
            if selected not in cards:
                selected = fallback
            self._previous_root = root
            self._previous_card = selected
            self.last_search_stats = diagnostics
            return selected
        except Exception:
            self._previous_root = None
            self._previous_card = fallback
            return fallback
