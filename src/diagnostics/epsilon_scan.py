from __future__ import annotations

from src.attacks.factory import build_attack_config
from src.attacks.runner import AttackRunner


def run_epsilon_scan(
    model,
    dataloader,
    device,
    eps_list: list[float],
    attack_name: str = "pgd20",
    steps: int = 20,
    max_samples: int = 0,
    max_eval_batches: int = 0,
) -> list[dict]:
    rows = []
    for eps in eps_list:
        result = AttackRunner(
            build_attack_config(attack_name, eps=eps, steps=steps),
            device=device,
            max_samples=max_samples,
            max_eval_batches=max_eval_batches,
        ).run(model, dataloader)
        rows.append({"eps": eps, "attack": attack_name, "robust_acc": result.get("robust_acc"), "status": result.get("status")})
    return rows


def is_monotone_nonincreasing(values: list[float | None], tolerance: float = 1e-4) -> bool:
    clean = [v for v in values if v is not None]
    return all(next_value <= value + tolerance for value, next_value in zip(clean, clean[1:]))
