from __future__ import annotations


def test_readme_declares_whitebox_scope() -> None:
    text = open("README.md", encoding="utf-8").read()
    assert "white-box only" in text
    assert "Square is evaluated separately" in text


def test_aggregate_preserves_r_lite_scope(tmp_path) -> None:
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path("scripts").resolve()))
    from scripts.aggregate_results import aggregate

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "aalite.json").write_text(
        json.dumps(
            {
                "exp_id": "x",
                "model": "smallcnn",
                "defense": "standard",
                "dataset_name": "cifar10",
                "mode": "real",
                "r_lite": 0.1,
                "gap_over": 0.2,
                "r_lite_scope": "whitebox",
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "tables"
    aggregate(raw, out)
    csv_text = (out / "main_robustness.csv").read_text(encoding="utf-8")
    assert "r_lite_scope" in csv_text
    assert "whitebox" in csv_text
