from pathlib import Path

import torch

try:
    from .rl_features import extract_policy_features
    from .rl_model import load_policy_model, select_card_from_logits
except ImportError:
    from rl_features import extract_policy_features
    from rl_model import load_policy_model, select_card_from_logits


HERE = Path(__file__).resolve().parent


class PolicyModelPlayer:
    """Loads a policy MLP and plays masked argmax or temperature sampling."""

    def __init__(self, player_idx: int, checkpoint: str = "policy_model.pth", temperature: float = 0.0, sample: bool = False):
        self.player_idx = int(player_idx)
        self.temperature = float(temperature)
        self.sample = bool(sample)
        self.model = load_policy_model(HERE / checkpoint, map_location="cpu")
        self.model.eval()

    def action(self, hand, history):
        if not hand:
            return 1
        cards = sorted(int(card) for card in hand)
        if len(cards) == 1:
            return cards[0]
        try:
            features, legal_mask = extract_policy_features(cards, history, self.player_idx)
            with torch.no_grad():
                x = torch.from_numpy(features).float().unsqueeze(0)
                legal = torch.from_numpy(legal_mask).bool()
                logits = self.model(x).squeeze(0)
                card = select_card_from_logits(logits, legal, temperature=self.temperature, sample=self.sample)
            return card if card in cards else cards[0]
        except Exception:
            return cards[0]
