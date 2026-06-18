from pathlib import Path

import torch

from .rl_model import PolicyMLP


def load_checkpoint(path, device="cpu"):
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = PolicyMLP().to(device)
    optimizer_state = None
    metadata = {}
    payload = torch.load(ckpt_path, map_location=device)
    if isinstance(payload, dict) and "model" in payload:
        model.load_state_dict(payload["model"])
        optimizer_state = payload.get("optimizer")
        metadata = payload.get("metadata", {})
    else:
        model.load_state_dict(payload)
    return model, optimizer_state, metadata


def save_checkpoint(path, model, optimizer=None, **metadata):
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "metadata": metadata}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, ckpt_path)


def clone_checkpoint(src, dst, extra_metadata=None):
    src = Path(src)
    dst = Path(dst)
    payload = torch.load(src, map_location="cpu")
    if isinstance(payload, dict):
        payload = dict(payload)
        metadata = dict(payload.get("metadata", {}))
        if extra_metadata:
            metadata.update(extra_metadata)
        payload["metadata"] = metadata
        payload.pop("optimizer", None)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, dst)
