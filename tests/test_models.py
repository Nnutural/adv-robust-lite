from __future__ import annotations

import importlib.util

import pytest

torch_available = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torchvision") is not None
pytestmark = pytest.mark.skipif(not torch_available, reason="torch/torchvision is not installed")


def test_model_forward_shapes() -> None:
    import torch

    from src.models.factory import build_model

    x = torch.rand(2, 3, 32, 32)
    for name in ("smallcnn", "resnet18", "mobilenetv2", "preact_resnet18"):
        model = build_model(name, normalize=True)
        logits = model(x)
        assert logits.shape == (2, 10)

