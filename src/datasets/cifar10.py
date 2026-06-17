from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .transforms import build_cifar10_transforms


@dataclass
class CIFAR10DataModule:
    root: str | Path = "data/raw/cifar10"
    processed_dir: str | Path = "data/processed"
    batch_size: int = 128
    num_workers: int = 4
    val_ratio: float = 0.1
    seed: int = 0
    download: bool = True
    aa_subset_size: int = 1000
    vis_subset_size: int = 64
    train_subset_size: int = 0
    val_subset_size: int = 0
    dataset_name: str = "cifar10"
    mode: str = "real"

    def setup(self) -> None:
        import torch
        from torch.utils.data import Subset
        from torchvision.datasets import CIFAR10, FakeData

        _validate_dataset_mode(self.dataset_name, self.mode)
        root = Path(self.root)
        processed_dir = Path(self.processed_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)

        train_transform = build_cifar10_transforms(train=True)
        eval_transform = build_cifar10_transforms(train=False)

        if self.dataset_name == "fake_cifar10":
            train_full = FakeData(
                size=500,
                image_size=(3, 32, 32),
                num_classes=10,
                transform=train_transform,
                random_offset=self.seed,
            )
            train_full_eval_transform = FakeData(
                size=500,
                image_size=(3, 32, 32),
                num_classes=10,
                transform=eval_transform,
                random_offset=self.seed,
            )
            test_full = FakeData(
                size=200,
                image_size=(3, 32, 32),
                num_classes=10,
                transform=eval_transform,
                random_offset=10_000 + self.seed,
            )
        elif self.dataset_name == "cifar10":
            train_full = CIFAR10(root=root, train=True, download=self.download, transform=train_transform)
            train_full_eval_transform = CIFAR10(root=root, train=True, download=False, transform=eval_transform)
            test_full = CIFAR10(root=root, train=False, download=self.download, transform=eval_transform)
        else:
            raise ValueError(f"Unsupported dataset_name: {self.dataset_name}")

        total_train = len(train_full)
        val_size = int(total_train * self.val_ratio)
        train_size = total_train - val_size

        generator = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(total_train, generator=generator).tolist()
        train_indices = perm[:train_size]
        val_indices = perm[train_size:]
        if self.train_subset_size:
            train_indices = train_indices[: min(self.train_subset_size, len(train_indices))]
        if self.val_subset_size:
            val_indices = val_indices[: min(self.val_subset_size, len(val_indices))]

        test_total = len(test_full)
        test_perm = torch.randperm(test_total, generator=torch.Generator().manual_seed(self.seed)).tolist()
        aa_subset_indices = test_perm[: min(self.aa_subset_size, test_total)]
        vis_subset_indices = test_perm[: min(self.vis_subset_size, test_total)]

        self._write_indices("train_indices", train_indices)
        self._write_indices("val_indices", val_indices)
        self._write_indices("aa_subset_indices", aa_subset_indices)
        self._write_indices("vis_subset_indices", vis_subset_indices)

        self.train_dataset = Subset(train_full, train_indices)
        self.val_dataset = Subset(train_full_eval_transform, val_indices)
        self.test_dataset = test_full
        self.aa_subset = Subset(test_full, aa_subset_indices)
        self.vis_subset = Subset(test_full, vis_subset_indices)

    def _write_indices(self, stem: str, indices: list[int]) -> None:
        path = Path(self.processed_dir) / f"{stem}_seed{self.seed}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(indices, handle)

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, shuffle=False)

    def aa_subset_dataloader(self):
        return self._loader(self.aa_subset, shuffle=False)

    def vis_subset_dataloader(self):
        return self._loader(self.vis_subset, shuffle=False)

    def _loader(self, dataset, shuffle: bool):
        from torch.utils.data import DataLoader

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=False,
        )


def build_dataloaders(cfg: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = cfg.get("dataset", {})
    project_cfg = cfg.get("project", {})
    mode = str(dataset_cfg.get("mode", cfg.get("mode", project_cfg.get("mode", "real"))))
    module = CIFAR10DataModule(
        root=dataset_cfg.get("root", "data/raw/cifar10"),
        processed_dir=dataset_cfg.get("processed_dir", "data/processed"),
        batch_size=int(dataset_cfg.get("batch_size", 128)),
        num_workers=int(dataset_cfg.get("num_workers", 4)),
        val_ratio=float(dataset_cfg.get("val_ratio", 0.1)),
        seed=int(project_cfg.get("seed", 0)),
        download=bool(dataset_cfg.get("download", True)),
        aa_subset_size=int(dataset_cfg.get("aa_subset_size", 1000)),
        vis_subset_size=int(dataset_cfg.get("vis_subset_size", 64)),
        train_subset_size=int(dataset_cfg.get("train_subset_size") or 0),
        val_subset_size=int(dataset_cfg.get("val_subset_size") or 0),
        dataset_name=dataset_cfg.get("name", "cifar10"),
        mode=mode,
    )
    module.setup()
    return {
        "train": module.train_dataloader(),
        "val": module.val_dataloader(),
        "test": module.test_dataloader(),
        "aa_subset": module.aa_subset_dataloader(),
        "vis_subset": module.vis_subset_dataloader(),
        "module": module,
    }


def build_cifar10_loaders(
    root: str | Path = "data/raw/cifar10",
    batch_size: int = 128,
    num_workers: int = 4,
    seed: int = 0,
    download: bool = True,
    val_ratio: float = 0.1,
    train_subset_size: int = 0,
    val_subset_size: int = 0,
    dataset_name: str = "cifar10",
    mode: str = "real",
):
    module = CIFAR10DataModule(
        root=root,
        processed_dir=Path(root).parent.parent / "processed" if Path(root).name == "cifar10" else "data/processed",
        batch_size=batch_size,
        num_workers=num_workers,
        val_ratio=val_ratio,
        seed=seed,
        download=download,
        train_subset_size=train_subset_size,
        val_subset_size=val_subset_size,
        dataset_name=dataset_name,
        mode=mode,
    )
    module.setup()
    return module.train_dataloader(), module.val_dataloader(), module.test_dataloader()


def build_cifar10_test_loader(
    root: str | Path = "data/raw/cifar10",
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = True,
    max_samples: int = 0,
    subset_size: int = 0,
    subset_indices_path: str | Path | None = None,
    seed: int = 0,
    dataset_name: str = "cifar10",
    mode: str = "real",
):
    from torch.utils.data import DataLoader, Subset
    from torchvision.datasets import CIFAR10, FakeData

    _validate_dataset_mode(dataset_name, mode)
    if dataset_name == "fake_cifar10":
        dataset = FakeData(
            size=200,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=build_cifar10_transforms(train=False),
            random_offset=10_000 + seed,
        )
    elif dataset_name == "cifar10":
        dataset = CIFAR10(root=Path(root), train=False, download=download, transform=build_cifar10_transforms(train=False))
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")
    limit = max_samples or subset_size
    if subset_indices_path is not None and Path(subset_indices_path).exists():
        with Path(subset_indices_path).open("r", encoding="utf-8") as handle:
            indices = json.load(handle)
        if limit:
            indices = indices[:limit]
        dataset = Subset(dataset, indices)
    elif limit:
        import torch

        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed)).tolist()[:limit]
        dataset = Subset(dataset, indices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)


def eval_subset_id(dataset_name: str, name: str, size: int, seed: int) -> str:
    return hashlib.sha1(f"{dataset_name}|{name}|{size}|{seed}".encode("utf-8")).hexdigest()[:12]


def build_named_eval_loader(
    name: str,
    subsets_cfg: dict[str, Any],
    root: str | Path = "data/raw/cifar10",
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = True,
    dataset_name: str = "cifar10",
    mode: str = "real",
):
    cfg = subsets_cfg.get("subsets", {}).get(name, {})
    size = int(cfg.get("size", 0) or 0)
    seed = int(cfg.get("seed", 0))
    loader = build_cifar10_test_loader(
        root=root,
        batch_size=batch_size,
        num_workers=num_workers,
        download=download,
        subset_size=size,
        seed=seed,
        dataset_name=dataset_name,
        mode=mode,
    )
    return loader, eval_subset_id(dataset_name, name, size, seed), list(range(size)) if size else []


def _validate_dataset_mode(dataset_name: str, mode: str) -> None:
    if mode not in {"real", "smoke"}:
        raise ValueError(f"Unsupported mode: {mode}")
    if mode == "real" and dataset_name == "fake_cifar10":
        raise RuntimeError("real mode forbids fake_cifar10; use --mode smoke for fake smoke runs.")
