from __future__ import annotations

import importlib.util

import pytest

torch_available = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not torch_available, reason="torch is not installed")


def test_fgsm_and_pgd_shape_and_range() -> None:
    import torch
    from torch import nn

    from src.attacks.factory import build_attack_config, AttackFactory

    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
    images = torch.rand(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3])
    for name in ("fgsm", "pgd20"):
        attack = AttackFactory.create(build_attack_config(name, steps=2))
        adv = attack(model, images, labels)
        assert adv.shape == images.shape
        assert float(adv.min()) >= 0.0
        assert float(adv.max()) <= 1.0


def test_apgd_missing_dependency_is_recorded_as_skipped() -> None:
    if importlib.util.find_spec("torchattacks") is not None:
        pytest.skip("torchattacks is installed; missing-dependency branch is not active")

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from src.attacks.factory import build_attack_config
    from src.attacks.runner import AttackRunner

    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
    dataset = TensorDataset(torch.rand(4, 3, 32, 32), torch.tensor([0, 1, 2, 3]))
    loader = DataLoader(dataset, batch_size=2)

    result = AttackRunner(
        build_attack_config("apgd_ce", steps=2),
        device="cpu",
        max_eval_batches=1,
    ).run(model, loader)

    assert result["status"] == "skipped"
    assert result["robust_acc"] is None
    assert result["attack_success_rate"] is None
    assert result["error"]


def test_runner_respects_max_eval_batches() -> None:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from src.attacks.factory import build_attack_config
    from src.attacks.runner import AttackRunner

    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
    dataset = TensorDataset(torch.rand(6, 3, 32, 32), torch.tensor([0, 1, 2, 3, 4, 5]))
    loader = DataLoader(dataset, batch_size=2)

    result = AttackRunner(
        build_attack_config("fgsm", steps=1),
        device="cpu",
        max_eval_batches=1,
    ).run(model, loader)

    assert result["status"] == "ok"
    assert result["num_samples"] == 2
    assert result["max_eval_batches"] == 1
