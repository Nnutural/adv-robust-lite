from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.utils.config import parse_float_or_fraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CIFAR-10 model with standard or adversarial training.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--model", default="smallcnn")
    parser.add_argument("--defense", choices=["standard", "fgsm_at", "pgd_at", "fixed_mixed_at", "budget_scheduler_at"], default="standard")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default="data/raw/cifar10")
    parser.add_argument("--dataset", choices=["cifar10", "fake_cifar10"], default="cifar10")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--train-subset-size", type=int, default=0)
    parser.add_argument("--val-subset-size", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--eps", type=parse_float_or_fraction, default=parse_float_or_fraction("8/255"))
    parser.add_argument("--pgd-steps", type=int, default=7)
    parser.add_argument("--pgd-alpha", type=parse_float_or_fraction, default=parse_float_or_fraction("2/255"))
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    from src.datasets.cifar10 import build_cifar10_loaders
    from src.models.factory import build_model
    from src.training.trainer import TrainConfig, Trainer
    from src.utils.seed import seed_everything

    seed_everything(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = build_cifar10_loaders(
        root=ROOT / args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        download=args.download,
        train_subset_size=args.train_subset_size,
        val_subset_size=args.val_subset_size,
        dataset_name=args.dataset,
    )
    model = build_model(args.model, normalize=True)
    config = TrainConfig(
        model_name=args.model,
        defense=args.defense,
        epochs=args.epochs,
        lr=args.lr,
        amp=args.amp,
        eps=args.eps,
        pgd_alpha=args.pgd_alpha,
        pgd_steps=args.pgd_steps,
        seed=args.seed,
        output_dir=str(ROOT / args.output_dir),
        run_name=args.run_name,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
    )
    metrics = Trainer(model, train_loader, val_loader, device, config).train()
    print(metrics)


if __name__ == "__main__":
    main()
