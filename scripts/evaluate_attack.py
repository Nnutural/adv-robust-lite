from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.utils.config import parse_float_or_fraction


def _requires_eot(wrappers) -> bool:
    return any((wrapper or {}).get("kind") == "input_random" for wrapper in wrappers or [])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one attack.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--attack", choices=["fgsm", "pgd20", "apgd_ce", "apgd_dlr", "square"], default="pgd20")
    parser.add_argument("--eps", type=parse_float_or_fraction, default=parse_float_or_fraction("8/255"))
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--n-queries", type=int, default=2000)
    parser.add_argument("--eot-samples", type=int, default=0)
    parser.add_argument("--model-wrappers", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default="data/raw/cifar10")
    parser.add_argument("--dataset", choices=["cifar10", "fake_cifar10"], default="cifar10")
    parser.add_argument("--mode", choices=["real", "smoke"], default="real")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--subset-size", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-subset", default=None)
    parser.add_argument("--subset-indices-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    from src.attacks.factory import build_attack_config
    from src.attacks.runner import AttackRunner
    from src.datasets.cifar10 import build_cifar10_test_loader, build_named_eval_loader, eval_subset_id
    from src.models.factory import apply_model_wrappers, build_model
    from src.utils.config import load_yaml
    from src.utils.checkpoint import extract_model_state, load_checkpoint

    sample_limit = args.subset_size or args.max_samples
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    wrappers = None
    if args.model_wrappers:
        wrapper_cfg = load_yaml(args.model_wrappers)
        wrappers = wrapper_cfg.get("eval", {}).get("model_wrappers") or wrapper_cfg.get("model", {}).get("wrappers")
    eot_required = _requires_eot(wrappers)
    model = build_model(args.model, normalize=True, wrappers=None)
    checkpoint = load_checkpoint(args.checkpoint, map_location=str(device))
    model.load_state_dict(extract_model_state(checkpoint), strict=False)
    model = apply_model_wrappers(model, wrappers).to(device)
    if args.eval_subset:
        subsets_cfg = load_yaml(ROOT / "configs/eval/subsets.yaml")
        loader, subset_id, _ = build_named_eval_loader(
            args.eval_subset,
            subsets_cfg,
            root=ROOT / args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            download=args.download,
            dataset_name=args.dataset,
            mode=args.mode,
        )
    else:
        subset_name = Path(args.subset_indices_path).stem if args.subset_indices_path else "legacy"
        subset_id = eval_subset_id(args.dataset, subset_name, sample_limit, args.seed)
        loader = build_cifar10_test_loader(
            root=ROOT / args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            download=args.download,
            subset_indices_path=ROOT / args.subset_indices_path if args.subset_indices_path else None,
            subset_size=sample_limit,
            seed=args.seed,
            dataset_name=args.dataset,
            mode=args.mode,
        )
    config = build_attack_config(args.attack, eps=args.eps, steps=args.steps, eot_samples=args.eot_samples, n_queries=args.n_queries)
    run_name = Path(args.checkpoint).parent.name
    output = args.output or ROOT / "results" / args.mode / "raw" / "attacks" / f"{run_name}_{args.attack}.json"
    per_class = Path(output).with_name(Path(output).stem + "_per_class.csv")
    metadata = {
        "exp_id": run_name,
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "checkpoint_path": str(args.checkpoint),
        "subset_size": sample_limit,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": str(device),
        "dataset": args.dataset,
        "dataset_name": args.dataset,
        "mode": args.mode,
        "seed": args.seed,
        "eot_samples": args.eot_samples,
        "eot_required": eot_required,
        "eot_disabled_for_demo": eot_required and args.eot_samples == 0,
        "eval_subset_id": subset_id,
    }
    result = AttackRunner(
        config,
        device=device,
        max_samples=sample_limit,
        max_eval_batches=args.max_eval_batches,
    ).run(model, loader, output_json=output, per_class_csv=per_class, metadata=metadata)
    print(result)


if __name__ == "__main__":
    main()
