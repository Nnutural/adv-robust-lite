from __future__ import annotations

from src.training.trainer import TrainConfig, Trainer


def train_pgd_at(model, train_loader, val_loader, device, **kwargs):
    config = TrainConfig(defense="pgd_at", **kwargs)
    return Trainer(model, train_loader, val_loader, device, config).train()

