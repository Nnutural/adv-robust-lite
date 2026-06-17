from __future__ import annotations

from pathlib import Path

from src.utils.config import load_yaml


def test_pgd_at_train_attack_is_weaker_than_eval_attack() -> None:
    cfg = load_yaml(Path("configs/defenses/pgd_at.yaml"))["training"]
    train_steps = int(cfg["train_attack"]["steps"])
    eval_attacks = cfg["eval"]["attacks"]
    assert "pgd20" in eval_attacks
    assert train_steps < 20


def test_train_attack_config_is_recorded_by_train_config() -> None:
    import pytest

    pytest.importorskip("torch")
    from src.training.trainer import TrainConfig

    train_attack = {"name": "pgd", "steps": 7, "eps": 8 / 255, "alpha": 2 / 255}
    cfg = TrainConfig(model_name="smallcnn", defense="pgd_at", train_attack=train_attack, eval_attacks=["pgd20"])
    assert cfg.train_attack["steps"] == 7
    assert cfg.eval_attacks == ["pgd20"]
