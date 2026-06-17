from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def test_budget_comparison_records_fairness_columns(tmp_path) -> None:
    sys.path.insert(0, str(Path("scripts").resolve()))
    from scripts.aggregate_results import aggregate

    raw = tmp_path / "raw"
    raw.mkdir()
    base = {
        "model": "smallcnn",
        "seed": 0,
        "dataset_name": "cifar10",
        "mode": "real",
        "clean_acc": 0.2,
        "r_lite": 0.1,
        "train_time_gpu_hours": 0.01,
        "gpu_name": "cpu",
        "amp": False,
        "session_id": "session-a",
        "max_wall_seconds": 30,
        "experiment_group": "g5_smoke",
        "r_lite_scope": "whitebox",
    }
    (raw / "standard.json").write_text(
        json.dumps({**base, "exp_id": "std", "defense": "standard", "batch_size": 128}),
        encoding="utf-8",
    )
    (raw / "pgd_at.json").write_text(
        json.dumps({**base, "exp_id": "pgd", "defense": "pgd_at", "batch_size": 64}),
        encoding="utf-8",
    )

    out = tmp_path / "tables"
    aggregate(raw, out)

    budget = pd.read_csv(out / "budget_comparison.csv")
    for column in ["gpu_name", "batch_size", "amp", "session_id", "max_wall_seconds", "equal_budget_invalid"]:
        assert column in budget.columns
    assert budget["equal_budget_invalid"].astype(str).str.lower().eq("true").all()
