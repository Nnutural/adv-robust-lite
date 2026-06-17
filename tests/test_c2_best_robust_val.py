from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from src.datasets.cifar10 import CIFAR10DataModule  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.training.trainer import TrainConfig, Trainer  # noqa: E402


def _module(tmp_path):
    module = CIFAR10DataModule(
        root=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        dataset_name="fake_cifar10",
        batch_size=8,
        num_workers=0,
        seed=0,
        mode="smoke",
        train_subset_size=16,
        val_subset_size=8,
        aa_subset_size=8,
        vis_subset_size=4,
    )
    module.setup()
    return module


def test_at_uses_robust_best_and_standard_uses_clean_best(tmp_path) -> None:
    module = _module(tmp_path)
    at_config = TrainConfig(
        model_name="smallcnn",
        defense="fgsm_at",
        epochs=1,
        lr=0.01,
        weight_decay=0.0,
        seed=0,
        output_dir=str(tmp_path / "checkpoints"),
        run_name="fgsm_at_case",
        max_train_batches=1,
        max_eval_batches=1,
        best_criterion="robust_val",
        robust_val_steps=1,
        robust_val_subset_size=8,
    )
    Trainer(build_model("smallcnn"), module.train_dataloader(), module.val_dataloader(), torch.device("cpu"), at_config).train()
    metrics = json.loads((tmp_path / "checkpoints" / "fgsm_at_case" / "metrics.json").read_text())
    assert metrics["best_criterion"] == "robust_val"
    assert metrics["best_metric"] == "pgd7_val_acc"
    assert "best_robust_acc" in metrics

    std_config = TrainConfig(
        model_name="smallcnn",
        defense="standard",
        epochs=1,
        lr=0.01,
        weight_decay=0.0,
        seed=0,
        output_dir=str(tmp_path / "checkpoints"),
        run_name="standard_case",
        max_train_batches=1,
        max_eval_batches=1,
        best_criterion="clean_val",
    )
    Trainer(build_model("smallcnn"), module.train_dataloader(), module.val_dataloader(), torch.device("cpu"), std_config).train()
    std_metrics = json.loads((tmp_path / "checkpoints" / "standard_case" / "metrics.json").read_text())
    assert std_metrics["best_criterion"] == "clean_val"
    assert std_metrics["best_metric"] == "val_acc"
