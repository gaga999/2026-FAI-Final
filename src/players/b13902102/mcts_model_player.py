from pathlib import Path
import random

try:
    from .rl_mcs import MCSConfig, root_mcs_search
    from .rl_model import load_policy_model
except ImportError:
    from rl_mcs import MCSConfig, root_mcs_search
    from rl_model import load_policy_model


HERE = Path(__file__).resolve().parent


class MCTSModelPlayer:
    """Loads a policy MLP and uses it as the prior for root PUCT/MCS."""

    def __init__(
        self,
        player_idx: int,
        checkpoint: str = "mcts_model.pth",
        sims: int = 48,
        c_puct: float = 1.4,
        root_temperature: float = 0.0,
        rollout_temperature: float = 0.7,
        deadline_s: float = 0.72,
        seed: int = 0,
    ):
        self.player_idx = int(player_idx)
        self.model = load_policy_model(HERE / checkpoint, map_location="cpu")
        self.model.eval()
        self.config = MCSConfig(
            sims=sims,
            c_puct=c_puct,
            root_temperature=root_temperature,
            rollout_temperature=rollout_temperature,
            deadline_s=min(0.90, max(0.05, float(deadline_s))),
        )
        self.rng = random.Random(int(seed) + 10007 * self.player_idx)

    def action(self, hand, history):
        if not hand:
            return 1
        cards = sorted(int(card) for card in hand)
        if len(cards) == 1:
            return cards[0]
        try:
            card, _, _ = root_mcs_search(self.model, cards, history, self.player_idx, self.config, self.rng)
            return card if card in cards else cards[0]
        except Exception:
            return cards[0]
