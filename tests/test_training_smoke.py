import pytest


torch = pytest.importorskip("torch")

from datasets.cifar10 import CIFAR10DataModule  # noqa: E402
from models.factory import build_model  # noqa: E402
from training.standard import train_standard  # noqa: E402


def test_standard_training_one_batch(tmp_path) -> None:
    module = CIFAR10DataModule(
        root=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        dataset_name="fake_cifar10",
        batch_size=8,
        num_workers=0,
        seed=0,
        mode="smoke",
        aa_subset_size=16,
        vis_subset_size=8,
    )
    module.setup()
    model = build_model("smallcnn", normalize=True)
    result = train_standard(
        model,
        module.train_dataloader(),
        module.val_dataloader(),
        torch.device("cpu"),
        model_name="smallcnn",
        epochs=1,
        lr=0.01,
        weight_decay=0.0,
        amp=False,
        seed=0,
        output_dir=str(tmp_path / "checkpoints"),
        run_name="pytest_smallcnn",
    )
    assert result["clean_acc"] >= 0.0
    assert (tmp_path / "checkpoints" / "pytest_smallcnn" / "best.pt").exists()
    assert (tmp_path / "checkpoints" / "pytest_smallcnn" / "metrics.json").exists()
