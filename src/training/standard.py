from __future__ import annotations

from src.training.trainer import TrainConfig, Trainer


def train_standard(model, train_loader, val_loader, device, **kwargs):
    config = TrainConfig(defense="standard", **kwargs)
    return Trainer(model, train_loader, val_loader, device, config).train()

