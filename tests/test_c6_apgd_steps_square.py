from __future__ import annotations

import importlib.util

from src.utils.config import load_yaml


def test_apgd_defaults_to_50_steps() -> None:
    import pytest

    pytest.importorskip("torch")
    from src.attacks.factory import build_attack_config

    assert build_attack_config("apgd_ce").steps == 50
    assert build_attack_config("apgd_dlr").steps == 50
    assert load_yaml("configs/attacks/apgd_ce.yaml")["attack"]["steps"] == 50
    assert load_yaml("configs/attacks/apgd_dlr.yaml")["attack"]["steps"] == 50


def test_square_config_defaults() -> None:
    import pytest

    pytest.importorskip("torch")
    from src.attacks.factory import build_attack_config

    cfg = build_attack_config("square")
    assert cfg.name == "square"
    assert cfg.n_queries == 2000
    assert load_yaml("configs/attacks/square.yaml")["attack"]["n_queries"] == 2000


def test_square_missing_package_is_skipped() -> None:
    import pytest

    torch = pytest.importorskip("torch")
    from src.attacks.factory import build_attack_config
    if importlib.util.find_spec("torchattacks") is not None:
        pytest.skip("torchattacks installed; missing dependency branch inactive")
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from src.attacks.runner import AttackRunner

    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
    loader = DataLoader(TensorDataset(torch.rand(2, 3, 32, 32), torch.tensor([0, 1])), batch_size=2)
    result = AttackRunner(build_attack_config("square"), device="cpu", max_eval_batches=1).run(model, loader)
    assert result["status"] == "skipped"
    assert "Square" in result["error"] or "torchattacks" in result["error"]
