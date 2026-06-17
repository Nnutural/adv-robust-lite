from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from src.attacks.factory import build_attack_config
from src.attacks.runner import AttackRunner
from src.evaluation.metrics import gap_over, r_lite
from src.utils.io import save_json

AA_LITE_ATTACKS = ("pgd20", "apgd_ce", "apgd_dlr")


def run_aalite(
    model,
    dataloader,
    device: torch.device | str,
    eps: float = 8 / 255,
    steps: int = 20,
    max_samples: int = 0,
    max_eval_batches: int = 0,
    output_json: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
    eot_samples: int = 0,
    eot_required: bool = False,
) -> dict[str, Any]:
    attack_results: dict[str, Any] = {}
    accs: dict[str, float] = {}
    clean_accs: list[float] = []
    errors: dict[str, str] = {}

    for attack_name in AA_LITE_ATTACKS:
        config = build_attack_config(attack_name, eps=eps, steps=steps, eot_samples=eot_samples)
        result = AttackRunner(
            config,
            device=device,
            max_samples=max_samples,
            max_eval_batches=max_eval_batches,
        ).run(model, dataloader, metadata=metadata)
        attack_results[attack_name] = result
        if result.get("status") == "ok" and result.get("robust_acc") is not None:
            key = "pgd20_acc" if attack_name == "pgd20" else f"{attack_name}_acc"
            accs[key] = float(result["robust_acc"])
            if result.get("clean_acc") is not None:
                clean_accs.append(float(result["clean_acc"]))
        else:
            errors[attack_name] = result.get("error", "unknown attack failure")

    successful = len(accs)
    if metadata and "eot_required" in metadata:
        eot_required = bool(metadata["eot_required"])
    payload: dict[str, Any] = {
        "eps": eps,
        "steps": steps,
        "num_samples": max((r.get("num_samples") or 0) for r in attack_results.values()) if attack_results else 0,
        "max_samples": max_samples,
        "max_eval_batches": max_eval_batches,
        "attacks": attack_results,
        "status": "ok" if not errors else ("partial" if successful else "failed"),
        "r_lite_scope": "whitebox",
        "blackbox_handled_separately": True,
        "blackbox_notes": "Square subset and black-box sanity diagnostics are evaluated separately.",
        "eot_samples": eot_samples,
        "eot_disabled_for_demo": eot_required and eot_samples == 0,
    }
    if metadata:
        payload.update(dict(metadata))
        payload["eot_disabled_for_demo"] = eot_required and eot_samples == 0
    if clean_accs and "clean_acc" not in payload:
        payload["clean_acc"] = clean_accs[0]
    payload.update(accs)
    subset_ids = {result.get("eval_subset_id") for result in attack_results.values() if result.get("eval_subset_id")}
    if metadata and metadata.get("eval_subset_id"):
        payload["eval_subset_id"] = metadata["eval_subset_id"]
    if len(subset_ids) > 1:
        payload["r_lite"] = None
        payload["gap_over"] = None
        payload["gap_over_error"] = "subset_id_mismatch"
        payload["errors"] = {**errors, "gap_over": "subset_id_mismatch"}
    elif {"pgd20_acc", "apgd_ce_acc", "apgd_dlr_acc"}.issubset(accs):
        value = r_lite(accs["pgd20_acc"], accs["apgd_ce_acc"], accs["apgd_dlr_acc"])
        payload["r_lite"] = value
        payload["gap_over"] = gap_over(accs["pgd20_acc"], value)
        payload["errors"] = {}
    else:
        payload["r_lite"] = None
        payload["gap_over"] = None
        payload["errors"] = errors

    if output_json:
        save_json(output_json, payload)
    return payload
