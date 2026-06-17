from __future__ import annotations

from typing import Mapping


def safe_div(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def compute_accuracy(correct: int | float, total: int | float) -> float:
    return safe_div(float(correct), float(total))


def accuracy_from_logits(logits, labels) -> float:
    preds = logits.argmax(dim=1)
    return float(preds.eq(labels).float().mean().item())


def r_lite(pgd20_acc: float, apgd_ce_acc: float, apgd_dlr_acc: float) -> float:
    return min(pgd20_acc, apgd_ce_acc, apgd_dlr_acc)


def gap_over(pgd20_acc: float, r_lite_acc: float) -> float:
    return pgd20_acc - r_lite_acc


def bias_aa(r_lite_subset: float, aa_subset_acc: float) -> float:
    return r_lite_subset - aa_subset_acc


def dev_aa(r_lite_subset: float, aa_subset_acc: float) -> float:
    return abs(bias_aa(r_lite_subset, aa_subset_acc))


def compute_aalite_metrics(
    pgd20_acc: float,
    apgd_ce_acc: float,
    apgd_dlr_acc: float,
) -> dict[str, float]:
    value = r_lite(pgd20_acc, apgd_ce_acc, apgd_dlr_acc)
    return {
        "pgd20_acc": pgd20_acc,
        "apgd_ce_acc": apgd_ce_acc,
        "apgd_dlr_acc": apgd_dlr_acc,
        "r_lite": value,
        "gap_over": gap_over(pgd20_acc, value),
    }


def compute_aa_subset_metrics(r_lite_subset: float, aa_subset_acc: float) -> dict[str, float]:
    bias_value = bias_aa(r_lite_subset, aa_subset_acc)
    return {
        "r_lite_subset": r_lite_subset,
        "aa_subset_acc": aa_subset_acc,
        "bias_aa": bias_value,
        "dev_aa": abs(bias_value),
    }


def merge_attack_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    required = ("pgd20_acc", "apgd_ce_acc", "apgd_dlr_acc")
    missing = [key for key in required if key not in metrics]
    if missing:
        raise KeyError(f"Missing AA-Lite attack metrics: {missing}")
    return compute_aalite_metrics(
        float(metrics["pgd20_acc"]),
        float(metrics["apgd_ce_acc"]),
        float(metrics["apgd_dlr_acc"]),
    )


def count_parameters(model, trainable_only: bool = True) -> int:
    params = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)
