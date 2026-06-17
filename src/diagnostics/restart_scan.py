from __future__ import annotations

from src.attacks.factory import build_attack_config
from src.attacks.runner import AttackRunner


def run_restart_scan(
    model,
    dataloader,
    device,
    restarts_list: list[int],
    eps: float = 8 / 255,
    steps: int = 20,
    max_samples: int = 0,
    max_eval_batches: int = 0,
) -> list[dict]:
    rows = []
    for restarts in restarts_list:
        config = build_attack_config("pgd20", eps=eps, steps=steps, restarts=restarts)
        result = AttackRunner(
            config,
            device=device,
            max_samples=max_samples,
            max_eval_batches=max_eval_batches,
        ).run(model, dataloader)
        rows.append(
            {
                "restarts": restarts,
                "robust_acc": result.get("robust_acc"),
                "status": result.get("status"),
            }
        )
    return rows


def restart_stable(values: list[float | None], max_extra_drop: float = 0.05) -> bool:
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return True
    return clean[0] - min(clean[1:]) <= max_extra_drop
