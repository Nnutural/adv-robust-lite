from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


def test_real_mode_forbids_fake_cifar10(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    from src.datasets.cifar10 import CIFAR10DataModule

    module = CIFAR10DataModule(
        root=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        dataset_name="fake_cifar10",
        mode="real",
        batch_size=2,
        num_workers=0,
    )
    with pytest.raises(RuntimeError, match="real mode forbids fake_cifar10"):
        module.setup()


def test_aggregate_defaults_to_real_and_rejects_smoke(tmp_path) -> None:
    sys.path.insert(0, str(Path("scripts").resolve()))
    from scripts.aggregate_results import aggregate, parse_args

    args = parse_args([])
    assert args.input == "results/real/raw"

    real_raw = tmp_path / "results" / "real" / "raw"
    real_raw.mkdir(parents=True)
    (real_raw / "smoke.json").write_text(
        json.dumps(
            {
                "exp_id": "smoke",
                "model": "smallcnn",
                "defense": "standard",
                "dataset_name": "fake_cifar10",
                "mode": "smoke",
                "r_lite": 0.1,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="refuses smoke/fake"):
        aggregate(real_raw, tmp_path / "tables")

