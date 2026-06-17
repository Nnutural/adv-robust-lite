from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _bootstrap import add_project_root

ROOT = add_project_root()

import pandas as pd

from src.evaluation.budget import clean_drop, gain_per_gpu_hour, robust_gain
from src.utils.io import ensure_dir


def _load_json_files(path: Path) -> list[dict[str, Any]]:
    rows = []
    for file in path.rglob("*.json"):
        try:
            with file.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError:
            continue
        payload["_source"] = str(file)
        rows.append(payload)
    return rows


def _flatten_aalite(rows: list[dict[str, Any]]) -> pd.DataFrame:
    records = []
    for payload in rows:
        source = Path(payload.get("_source", ""))
        exp_id = payload.get("exp_id") or source.stem
        record = {
            "exp_id": exp_id,
            "model": payload.get("model", ""),
            "defense": payload.get("defense", ""),
            "seed": payload.get("seed", ""),
            "clean_acc": payload.get("clean_acc", ""),
            "fgsm_acc": payload.get("fgsm_acc", ""),
            "pgd20_acc": payload.get("pgd20_acc", ""),
            "apgd_ce_acc": payload.get("apgd_ce_acc", ""),
            "apgd_dlr_acc": payload.get("apgd_dlr_acc", ""),
            "r_lite": payload.get("r_lite", ""),
            "gap_over": payload.get("gap_over", ""),
            "train_time_sec": payload.get("train_time_sec", ""),
            "train_time_gpu_hours": payload.get("train_time_gpu_hours", ""),
            "checkpoint_path": payload.get("checkpoint_path", ""),
            "status": payload.get("status", ""),
            "errors": json.dumps(payload.get("errors", {}), ensure_ascii=False) if payload.get("errors") else "",
            "subset_size": payload.get("subset_size", payload.get("num_samples", "")),
            "max_eval_batches": payload.get("max_eval_batches", ""),
            "batch_size": payload.get("batch_size", ""),
            "num_workers": payload.get("num_workers", ""),
        }
        attacks = payload.get("attacks")
        if isinstance(attacks, dict):
            for attack_name, result in attacks.items():
                key = "pgd20_acc" if attack_name == "pgd20" else f"{attack_name}_acc"
                if result.get("status") == "ok":
                    record[key] = result.get("robust_acc")
                else:
                    record[key] = ""
                    record[f"{attack_name}_status"] = result.get("status", "")
                    record[f"{attack_name}_error"] = result.get("error", "")
        records.append(record)
    return pd.DataFrame.from_records(records)


def aggregate(input_dir: Path, output_dir: Path) -> dict[str, str]:
    ensure_dir(output_dir)
    json_rows = _load_json_files(input_dir)
    aalite_rows = [row for row in json_rows if "r_lite" in row or "attacks" in row]
    aa_subset_rows = [row for row in json_rows if "aa_subset_acc" in row]
    diagnostics_rows = [row for row in json_rows if "diagnosis_pass" in row]

    outputs: dict[str, str] = {}
    main = _flatten_aalite(aalite_rows)
    main_path = output_dir / "main_robustness.csv"
    main.to_csv(main_path, index=False)
    outputs["main_robustness"] = str(main_path)

    aa_df = pd.DataFrame.from_records(aa_subset_rows)
    if aa_df.empty:
        aa_df = pd.DataFrame(
            columns=["exp_id", "model", "defense", "r_lite_subset", "aa_subset_acc", "bias_aa", "dev_aa", "square_acc", "subset_size"]
        )
    aa_path = output_dir / "aa_subset_check.csv"
    aa_df.to_csv(aa_path, index=False)
    outputs["aa_subset_check"] = str(aa_path)

    diag_records = []
    for row in diagnostics_rows:
        diag_records.append(
            {
                "exp_id": Path(row.get("_source", "diagnostics")).stem,
                "eps_monotone": row.get("eps_monotone"),
                "step_monotone": row.get("step_monotone"),
                "restart_stable": row.get("restart_stable"),
                "blackbox_not_stronger": row.get("blackbox_not_stronger"),
                "diagnosis_pass": row.get("diagnosis_pass"),
                "notes": row.get("notes", ""),
            }
        )
    diag_path = output_dir / "gradient_masking_diagnostics.csv"
    diag_df = pd.DataFrame.from_records(diag_records)
    if diag_df.empty:
        diag_df = pd.DataFrame(
            columns=["exp_id", "eps_monotone", "step_monotone", "restart_stable", "blackbox_not_stronger", "diagnosis_pass", "notes"]
        )
    diag_df.to_csv(diag_path, index=False)
    outputs["gradient_masking_diagnostics"] = str(diag_path)

    budget_rows = []
    if not main.empty and "defense" in main.columns:
        standard = main[main["defense"].fillna("") == "standard"]
        baseline = float(standard.iloc[0]["r_lite"]) if not standard.empty and standard.iloc[0].get("r_lite", "") != "" else None
        baseline_clean = float(standard.iloc[0]["clean_acc"]) if not standard.empty and standard.iloc[0].get("clean_acc", "") != "" else None
        for _, row in main.iterrows():
            r_value = row.get("r_lite", "")
            clean_value = row.get("clean_acc", "")
            gpu_hours = row.get("train_time_gpu_hours", 0) or 0
            try:
                r_float = float(r_value)
                clean_float = float(clean_value)
                gpu_float = float(gpu_hours)
            except (TypeError, ValueError):
                continue
            budget_rows.append(
                {
                    "defense": row.get("defense", ""),
                    "epoch_budget": "",
                    "gpu_hour_budget": gpu_float,
                    "clean_acc": clean_float,
                    "r_lite": r_float,
                    "clean_drop": clean_drop(baseline_clean, clean_float) if baseline_clean is not None else "",
                    "robust_gain": robust_gain(r_float, baseline) if baseline is not None else "",
                    "gain_per_gpu_hour": gain_per_gpu_hour(r_float, baseline, gpu_float) if baseline is not None else "",
                }
            )
    budget_path = output_dir / "budget_comparison.csv"
    pd.DataFrame.from_records(budget_rows).to_csv(budget_path, index=False)
    outputs["budget_comparison"] = str(budget_path)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate raw JSON results into tables.")
    parser.add_argument("--input", default="results/raw")
    parser.add_argument("--output", default="results/tables")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = aggregate(ROOT / args.input, ROOT / args.output)
    print(outputs)


if __name__ == "__main__":
    main()
