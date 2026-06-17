from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_fgsm_rs_uses_random_start_and_clamps() -> None:
    import torch
    from torch import nn

    from src.attacks.factory import fgsm_attack, fgsm_rs_attack

    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
    images = torch.rand(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3])
    fgsm = fgsm_attack(model, images, labels, eps=8 / 255)
    fgsm_rs = fgsm_rs_attack(model, images, labels, eps=8 / 255, alpha=10 / 255, random_start=True)
    assert fgsm_rs.shape == images.shape
    assert float(fgsm_rs.min()) >= 0.0
    assert float(fgsm_rs.max()) <= 1.0
    assert not torch.allclose(fgsm, fgsm_rs)


def test_co_drop_rule_detects_large_single_epoch_drop() -> None:
    history = [0.40, 0.42, 0.10]
    threshold = 0.15
    drops = [previous - current for previous, current in zip(history, history[1:])]
    assert any(drop > threshold for drop in drops)


def test_fgsm_at_config_uses_fgsm_rs_and_co_check() -> None:
    from src.utils.config import load_yaml

    cfg = load_yaml("configs/defenses/fgsm_at.yaml")["training"]
    assert cfg["train_attack"]["name"] == "fgsm_rs"
    assert cfg["co_check"]["enabled"] is True
