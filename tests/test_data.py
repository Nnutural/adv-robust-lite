from __future__ import annotations

import importlib.util

import pytest

torch_available = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torchvision") is not None
pytestmark = pytest.mark.skipif(not torch_available, reason="torch/torchvision is not installed")


def test_eval_transform_range() -> None:
    import numpy as np
    from PIL import Image

    from src.datasets.transforms import build_eval_transform

    image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    tensor = build_eval_transform()(image)
    assert tensor.shape == (3, 32, 32)
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0

