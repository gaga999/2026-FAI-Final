from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rl_features import FEATURE_DIM


class PolicyMLP(nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM, hidden_dims=(512, 256, 256), output_dim: int = 104):
        super().__init__()
        layers = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.LayerNorm(hidden))
            layers.append(nn.SiLU())
            prev = hidden
        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(prev, output_dim)

    def forward(self, features):
        return self.policy_head(self.backbone(features))


def mask_logits(logits, legal_mask):
    if legal_mask.dtype != torch.bool:
        legal_mask = legal_mask.bool()
    return logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)


def masked_policy(logits, legal_mask, temperature: float = 1.0):
    temperature = max(float(temperature), 1.0e-6)
    return F.softmax(mask_logits(logits, legal_mask) / temperature, dim=-1)


def select_card_from_logits(logits, legal_mask, temperature: float = 0.0, sample: bool = False):
    masked = mask_logits(logits, legal_mask)
    if sample or temperature > 0.0:
        probs = F.softmax(masked / max(float(temperature), 1.0e-6), dim=-1)
        idx = torch.multinomial(probs, num_samples=1).item()
    else:
        idx = int(torch.argmax(masked).item())
    return idx + 1


def policy_loss(logits, legal_mask, target_pi, entropy_coef: float = 0.0):
    masked = mask_logits(logits, legal_mask)
    log_probs = F.log_softmax(masked, dim=-1)
    probs = torch.exp(log_probs)
    target = target_pi / target_pi.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    ce = -(target * log_probs).sum(dim=-1).mean()
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    return ce - entropy_coef * entropy, {"ce": ce.detach(), "entropy": entropy.detach()}


def supervised_ce_loss(logits, legal_mask, action_indices, label_smoothing: float = 0.02, entropy_coef: float = 0.0):
    masked = mask_logits(logits, legal_mask)
    log_probs = F.log_softmax(masked, dim=-1)
    legal_float = legal_mask.float()
    legal_count = legal_float.sum(dim=-1, keepdim=True).clamp_min(1.0)
    smooth_target = legal_float / legal_count
    hard = F.one_hot(action_indices, num_classes=masked.shape[-1]).float()
    target = (1.0 - label_smoothing) * hard + label_smoothing * smooth_target
    ce = -(target * log_probs).sum(dim=-1).mean()
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    return ce - entropy_coef * entropy, {"ce": ce.detach(), "entropy": entropy.detach()}


def load_policy_model(checkpoint_path, map_location="cpu"):
    path = Path(checkpoint_path)
    model = PolicyMLP()
    if path.exists():
        ckpt = torch.load(path, map_location=map_location)
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state)
    model.eval()
    return model


def save_policy_checkpoint(path, model, optimizer=None, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "metadata": metadata}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)
