from __future__ import annotations

import torch
from torch import nn


class PolicyValueNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, n_cards: int = 104):
        super().__init__()
        self.n_cards = n_cards

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, n_cards)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        policy_logits = self.policy_head(h)
        value = torch.tanh(self.value_head(h)).squeeze(-1)
        return policy_logits, value


def masked_policy_distribution(
    logits: torch.Tensor,
    legal_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Converts policy logits to a legal-action probability distribution."""
    if logits.dim() != 1:
        raise ValueError("logits must be a 1-D tensor")
    if legal_mask.dim() != 1:
        raise ValueError("legal_mask must be a 1-D tensor")

    temp = max(1e-6, float(temperature))
    scaled = logits / temp

    masked = scaled.clone()
    masked[legal_mask <= 0.0] = -1e9

    probs = torch.softmax(masked, dim=0)

    # In case all legal actions were masked out due to malformed input.
    if torch.isnan(probs).any() or float(torch.sum(probs)) <= 0.0:
        fallback = legal_mask.clone()
        s = float(torch.sum(fallback))
        if s <= 0.0:
            fallback = torch.ones_like(legal_mask)
            s = float(torch.sum(fallback))
        probs = fallback / s

    return probs
