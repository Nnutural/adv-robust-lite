from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.evaluation.metrics import bias_aa, dev_aa
from src.utils.io import save_json


def run_autoattack_subset(
    model,
    dataloader,
    device: torch.device | str,
    eps: float = 8 / 255,
    r_lite_subset: float | None = None,
    max_eval_batches: int = 0,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    device = torch.device(device)
    model.to(device)
    model.eval()
    try:
        from autoattack import AutoAttack
    except ModuleNotFoundError as exc:
        payload = {
            "status": "skipped",
            "error": f"autoattack package is not installed; skipped official AutoAttack subset: {exc}",
            "eps": eps,
            "r_lite_subset": r_lite_subset,
            "aa_subset_acc": None,
            "bias_aa": None,
            "dev_aa": None,
            "subset_size": 0,
            "max_eval_batches": max_eval_batches,
        }
        if output_json:
            save_json(output_json, payload)
        return payload

    xs = []
    ys = []
    for batch_idx, (images, labels) in enumerate(dataloader):
        if max_eval_batches and batch_idx >= max_eval_batches:
            break
        xs.append(images)
        ys.append(labels)
    if not xs:
        payload = {
            "status": "failed",
            "error": "AutoAttack subset dataloader produced no samples.",
            "eps": eps,
            "r_lite_subset": r_lite_subset,
            "aa_subset_acc": None,
            "bias_aa": None,
            "dev_aa": None,
            "subset_size": 0,
            "max_eval_batches": max_eval_batches,
        }
        if output_json:
            save_json(output_json, payload)
        return payload
    x_test = torch.cat(xs, dim=0).to(device)
    y_test = torch.cat(ys, dim=0).to(device)
    try:
        adversary = AutoAttack(model, norm="Linf", eps=eps, version="standard", device=str(device), verbose=False)
        x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=dataloader.batch_size or 128)
    except Exception as exc:
        payload = {
            "status": "failed",
            "error": f"official AutoAttack subset failed: {exc}",
            "eps": eps,
            "r_lite_subset": r_lite_subset,
            "aa_subset_acc": None,
            "bias_aa": None,
            "dev_aa": None,
            "subset_size": int(y_test.numel()),
            "max_eval_batches": max_eval_batches,
        }
        if output_json:
            save_json(output_json, payload)
        return payload
    with torch.no_grad():
        preds = model(x_adv).argmax(dim=1)
        aa_subset_acc = (preds == y_test).float().mean().item()
    payload = {
        "status": "ok",
        "eps": eps,
        "r_lite_subset": r_lite_subset,
        "aa_subset_acc": aa_subset_acc,
        "bias_aa": bias_aa(r_lite_subset, aa_subset_acc) if r_lite_subset is not None else None,
        "dev_aa": dev_aa(r_lite_subset, aa_subset_acc) if r_lite_subset is not None else None,
        "subset_size": int(y_test.numel()),
        "max_eval_batches": max_eval_batches,
    }
    if output_json:
        save_json(output_json, payload)
    return payload
