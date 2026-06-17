from __future__ import annotations

from src.training.trainer import TrainConfig, Trainer


def train_fixed_mixed_at(model, train_loader, val_loader, device, **kwargs):
    config = TrainConfig(defense="fixed_mixed_at", **kwargs)
    return Trainer(model, train_loader, val_loader, device, config).train()

