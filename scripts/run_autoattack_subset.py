from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.utils.config import parse_float_or_fraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full AutoAttack on a fixed test subset.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--subset-size", type=int, default=1000)
    parser.add_argument("--eps", type=parse_float_or_fraction, default=parse_float_or_fraction("8/255"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default="data/raw/cifar10")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    from src.datasets.cifar10 import build_cifar10_test_loader
    from src.evaluation.autoattack_subset import run_autoattack_subset
    from src.models.factory import build_model
    from src.utils.checkpoint import extract_model_state, load_checkpoint

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = build_model(args.model, normalize=True).to(device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=str(device))
    model.load_state_dict(extract_model_state(checkpoint), strict=False)
    subset_path = ROOT / "data/processed" / f"aa_subset_indices_seed{args.seed}.json"
    loader = build_cifar10_test_loader(
        root=ROOT / args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
        subset_size=args.subset_size,
        seed=args.seed,
        subset_indices_path=subset_path,
    )
    run_name = Path(args.checkpoint).parent.name
    output = args.output or ROOT / "results/raw/autoattack_subset" / f"{run_name}.json"
    result = run_autoattack_subset(
        model,
        loader,
        device=device,
        eps=args.eps,
        max_eval_batches=args.max_eval_batches,
        output_json=output,
    )
    print(result)


if __name__ == "__main__":
    main()
