from __future__ import annotations

import importlib.util

import pytest

torch_available = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not torch_available, reason="torch is not installed")


def test_logit_scale_wrapper_multiplies_logits() -> None:
    import torch
    from torch import nn

    from src.models.trap_wrappers import LogitScaleWrapper

    base = nn.Linear(4, 2)
    wrapped = LogitScaleWrapper(base, scale=50.0)
    x = torch.rand(3, 4)
    assert torch.allclose(wrapped(x), base(x) * 50.0)


def test_input_randomization_wrapper_changes_forward() -> None:
    import torch
    from torch import nn

    from src.models.trap_wrappers import InputRandomizationWrapper

    class IdentityBackbone(nn.Module):
        def forward(self, x):
            return x

    wrapped = InputRandomizationWrapper(IdentityBackbone(), kind="pad_resize", pad=4, out_size=32)
    x = torch.linspace(0, 1, steps=3 * 32 * 32).view(1, 3, 32, 32)
    outputs = [wrapped(x) for _ in range(6)]
    assert any(not torch.equal(outputs[0], other) for other in outputs[1:])
    assert all(float(out.min()) >= 0.0 and float(out.max()) <= 1.0 for out in outputs)


def test_pgd_eot_path_runs_and_records_metadata() -> None:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from src.attacks.factory import build_attack_config
    from src.attacks.runner import AttackRunner

    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
    dataset = TensorDataset(torch.rand(4, 3, 32, 32), torch.tensor([0, 1, 2, 3]))
    loader = DataLoader(dataset, batch_size=2)
    result = AttackRunner(
        build_attack_config("pgd20", steps=2, eot_samples=4),
        device="cpu",
        max_eval_batches=1,
    ).run(model, loader)
    assert result["status"] == "ok"
    assert result["eot_samples"] == 4
    assert result["eot_disabled_for_demo"] is False


def test_aalite_payload_declares_whitebox_scope() -> None:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from src.attacks.aalite import run_aalite

    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
    dataset = TensorDataset(torch.rand(2, 3, 32, 32), torch.tensor([0, 1]))
    result = run_aalite(model, DataLoader(dataset, batch_size=2), device="cpu", steps=2, max_eval_batches=1)
    assert result["r_lite_scope"] == "whitebox"
    assert result["blackbox_handled_separately"] is True
