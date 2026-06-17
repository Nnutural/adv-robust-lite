from __future__ import annotations

from pathlib import Path
from typing import Any


def save_checkpoint(
    path: str | Path,
    model=None,
    optimizer=None,
    scheduler=None,
    epoch: int = 0,
    metrics: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model, dict) and optimizer is None and scheduler is None and metrics is None and config is None:
        payload = model
    else:
        if model is None:
            raise ValueError("model or checkpoint payload is required")
        payload: dict[str, Any] = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "metrics": metrics or {},
            "config": config or {},
        }
        if optimizer is not None:
            payload["optimizer_state"] = optimizer.state_dict()
        if scheduler is not None:
            payload["scheduler_state"] = scheduler.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | None = "cpu") -> dict[str, Any]:
    import torch

    return torch.load(Path(path), map_location=map_location)


def load_model_state(model, checkpoint: dict[str, Any]):
    state = extract_model_state(checkpoint)
    model.load_state_dict(state)
    return model


def extract_model_state(checkpoint: dict[str, Any]) -> dict[str, Any]:
    if "model_state" in checkpoint:
        return checkpoint["model_state"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint
