from __future__ import annotations

from pathlib import Path
from typing import Any

from src.diagnostics.epsilon_scan import is_monotone_nonincreasing, run_epsilon_scan
from src.diagnostics.restart_scan import restart_stable, run_restart_scan
from src.utils.io import save_json, write_csv


def run_step_scan(
    model,
    dataloader,
    device,
    steps_list: list[int],
    eps: float = 8 / 255,
    max_samples: int = 0,
    max_eval_batches: int = 0,
) -> list[dict]:
    from src.attacks.factory import build_attack_config
    from src.attacks.runner import AttackRunner

    rows = []
    for steps in steps_list:
        result = AttackRunner(
            build_attack_config("pgd20", eps=eps, steps=steps),
            device=device,
            max_samples=max_samples,
            max_eval_batches=max_eval_batches,
        ).run(model, dataloader)
        rows.append({"steps": steps, "robust_acc": result.get("robust_acc"), "status": result.get("status")})
    return rows


def run_gradient_masking_diagnostics(
    model,
    dataloader,
    device,
    eps_list: list[float],
    steps_list: list[int],
    restarts_list: list[int],
    max_samples: int = 0,
    max_eval_batches: int = 0,
    output_json: str | Path | None = None,
    output_csv: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    eps_rows = run_epsilon_scan(
        model,
        dataloader,
        device,
        eps_list,
        max_samples=max_samples,
        max_eval_batches=max_eval_batches,
    )
    step_rows = run_step_scan(
        model,
        dataloader,
        device,
        steps_list,
        eps=max(eps_list),
        max_samples=max_samples,
        max_eval_batches=max_eval_batches,
    )
    restart_rows = run_restart_scan(
        model,
        dataloader,
        device,
        restarts_list,
        eps=max(eps_list),
        steps=max(steps_list),
        max_samples=max_samples,
        max_eval_batches=max_eval_batches,
    )

    eps_pass = is_monotone_nonincreasing([row.get("robust_acc") for row in eps_rows])
    step_pass = is_monotone_nonincreasing([row.get("robust_acc") for row in step_rows])
    restart_pass = restart_stable([row.get("robust_acc") for row in restart_rows])
    payload = {
        **(metadata or {}),
        "eps_scan": eps_rows,
        "step_scan": step_rows,
        "restart_scan": restart_rows,
        "eps_monotone": eps_pass,
        "step_monotone": step_pass,
        "restart_stable": restart_pass,
        "max_samples": max_samples,
        "max_eval_batches": max_eval_batches,
        "blackbox_not_stronger": None,
        "diagnosis_pass": eps_pass and step_pass and restart_pass,
        "notes": "Black-box sanity is reserved unless MI-FGSM/Square results are supplied.",
    }
    if output_json:
        save_json(output_json, payload)
    if output_csv:
        write_csv(
            output_csv,
            [
                {
                    "exp_id": Path(output_json).stem if output_json else "diagnostics",
                    "dataset_name": (metadata or {}).get("dataset_name", ""),
                    "mode": (metadata or {}).get("mode", ""),
                    "eps_monotone": eps_pass,
                    "step_monotone": step_pass,
                    "restart_stable": restart_pass,
                    "blackbox_not_stronger": "",
                    "diagnosis_pass": payload["diagnosis_pass"],
                    "notes": payload["notes"],
                }
            ],
        )
    return payload
