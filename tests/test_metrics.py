from __future__ import annotations

from src.evaluation.metrics import bias_aa, compute_aa_subset_metrics, compute_aalite_metrics, dev_aa, gap_over, r_lite


def test_aalite_metrics() -> None:
    value = r_lite(0.62, 0.58, 0.55)
    assert value == 0.55
    assert abs(gap_over(0.62, value) - 0.07) < 1e-9
    assert abs(bias_aa(value, 0.52) - 0.03) < 1e-9
    assert abs(dev_aa(value, 0.52) - 0.03) < 1e-9


def test_p5_reference_metric_values() -> None:
    aalite = compute_aalite_metrics(
        pgd20_acc=0.40,
        apgd_ce_acc=0.35,
        apgd_dlr_acc=0.32,
    )
    assert aalite["r_lite"] == 0.32
    assert abs(aalite["gap_over"] - 0.08) < 1e-9

    aa_subset = compute_aa_subset_metrics(r_lite_subset=aalite["r_lite"], aa_subset_acc=0.30)
    assert abs(aa_subset["bias_aa"] - 0.02) < 1e-9
    assert abs(aa_subset["dev_aa"] - 0.02) < 1e-9
