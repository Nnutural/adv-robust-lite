from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _bootstrap import add_project_root

ROOT = add_project_root()

import pandas as pd

from src.evaluation.budget import clean_drop, gain_per_gpu_hour, robust_gain
from src.evaluation.metrics import bias_aa, dev_aa
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


def _load_training_metrics(path: Path) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for file in path.glob("*/metrics.json"):
        try:
            with file.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError:
            continue
        exp_id = payload.get("exp_id") or file.parent.name
        metrics[str(exp_id)] = payload
    return metrics


def _source_exp_id(payload: dict[str, Any]) -> str:
    source = Path(payload.get("_source", ""))
    return str(payload.get("exp_id") or source.stem)


def _file_exp_id(payload: dict[str, Any]) -> str:
    source = Path(payload.get("_source", ""))
    return source.stem or str(payload.get("exp_id", ""))


def _attack_match_ids(payload: dict[str, Any]) -> set[str]:
    ids = {_file_exp_id(payload)}
    attack = _attack_name(payload)
    file_id = _file_exp_id(payload)
    suffix = f"_{attack}"
    if attack and file_id.endswith(suffix):
        ids.add(file_id[: -len(suffix)])
    if not any(ids):
        ids.add(_source_exp_id(payload))
    return {value for value in ids if value}


def _actual_subset_size(payload: dict[str, Any]) -> Any:
    for key in ("num_samples", "subset_size", "max_samples"):
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return ""


def _is_ok(payload: dict[str, Any]) -> bool:
    return payload.get("status", "ok") == "ok"


def _attack_name(payload: dict[str, Any]) -> str:
    name = payload.get("attack") or payload.get("name") or ""
    return str(name).lower()


def _prefer_attack(current: dict[str, Any] | None, candidate: dict[str, Any]) -> dict[str, Any]:
    if current is None:
        return candidate
    current_size = int(current.get("num_samples") or current.get("subset_size") or 0)
    candidate_size = int(candidate.get("num_samples") or candidate.get("subset_size") or 0)
    attack = _attack_name(candidate)
    if attack == "pgd20":
        if candidate_size == 5000 and current_size != 5000:
            return candidate
        if current_size == 5000 and candidate_size != 5000:
            return current
    return candidate if candidate_size > current_size else current


def _standalone_attack_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    attacks: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if not _is_ok(row):
            continue
        if row.get("robust_acc") is None:
            continue
        attack = _attack_name(row)
        if attack not in {"fgsm", "pgd20", "square"}:
            continue
        for exp_id in _attack_match_ids(row):
            attacks.setdefault(exp_id, {})
            attacks[exp_id][attack] = _prefer_attack(attacks[exp_id].get(attack), row)
    return attacks


def _rows_from_subdir_or_filter(input_dir: Path, subdir: str, rows: list[dict[str, Any]], predicate) -> list[dict[str, Any]]:
    child = input_dir / subdir
    if child.exists():
        return _load_json_files(child)
    return [row for row in rows if predicate(row)]


def _reject_smoke_or_fake(rows: list[dict[str, Any]]) -> None:
    offenders: list[str] = []
    for row in rows:
        dataset_name = row.get("dataset_name", row.get("dataset", ""))
        mode = row.get("mode", "")
        if mode == "smoke" or dataset_name == "fake_cifar10":
            offenders.append(row.get("_source", "<unknown>"))
    if offenders:
        joined = "\n".join(offenders[:20])
        raise RuntimeError(f"real aggregation refuses smoke/fake results:\n{joined}")


def _flatten_aalite(
    rows: list[dict[str, Any]],
    training_metrics: dict[str, dict[str, Any]] | None = None,
    standalone_attacks: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> pd.DataFrame:
    records = []
    for payload in rows:
        exp_id = _file_exp_id(payload)
        train_meta = (training_metrics or {}).get(str(exp_id), {})
        attacks = payload.get("attacks")
        aalite_pgd20_acc = payload.get("pgd20_acc", "")
        if isinstance(attacks, dict):
            pgd20_result = attacks.get("pgd20", {})
            if isinstance(pgd20_result, dict) and pgd20_result.get("status") == "ok":
                aalite_pgd20_acc = pgd20_result.get("robust_acc", aalite_pgd20_acc)
        standalone = (standalone_attacks or {}).get(str(exp_id), {})
        if not standalone and str(exp_id).endswith("_eot"):
            standalone = (standalone_attacks or {}).get(str(exp_id)[: -len("_eot")], {})
        fgsm = standalone.get("fgsm")
        pgd20 = standalone.get("pgd20")
        square = standalone.get("square")
        pgd20_acc = pgd20.get("robust_acc") if pgd20 else aalite_pgd20_acc
        pgd20_source = ""
        if pgd20:
            pgd20_source = "standalone_attack"
        elif aalite_pgd20_acc != "":
            pgd20_source = "aalite_internal_1k"
        record = {
            "exp_id": exp_id,
            "model": payload.get("model", ""),
            "defense": train_meta.get("defense", payload.get("defense", "")),
            "seed": train_meta.get("seed", payload.get("seed", "")),
            "dataset_name": payload.get("dataset_name", payload.get("dataset", "")),
            "mode": payload.get("mode", ""),
            "clean_acc": payload.get("clean_acc", ""),
            "fgsm_acc": fgsm.get("robust_acc") if fgsm else payload.get("fgsm_acc", ""),
            "pgd20_acc": pgd20_acc,
            "pgd20_aalite_acc": aalite_pgd20_acc,
            "pgd20_source": pgd20_source,
            "pgd20_subset_size": _actual_subset_size(pgd20) if pgd20 else _actual_subset_size(payload),
            "fgsm_subset_size": _actual_subset_size(fgsm) if fgsm else (_actual_subset_size(payload) if payload.get("fgsm_acc", "") != "" else ""),
            "square_acc": square.get("robust_acc") if square else payload.get("square_acc", ""),
            "square_subset_size": _actual_subset_size(square) if square else "",
            "apgd_ce_acc": payload.get("apgd_ce_acc", ""),
            "apgd_dlr_acc": payload.get("apgd_dlr_acc", ""),
            "r_lite": payload.get("r_lite", ""),
            "gap_over": payload.get("gap_over", ""),
            "train_time_sec": train_meta.get("train_time_sec", payload.get("train_time_sec", "")),
            "train_time_gpu_hours": train_meta.get("train_time_gpu_hours", payload.get("train_time_gpu_hours", "")),
            "checkpoint_path": payload.get("checkpoint_path", ""),
            "status": payload.get("status", ""),
            "errors": json.dumps(payload.get("errors", {}), ensure_ascii=False) if payload.get("errors") else "",
            "subset_size": payload.get("subset_size", payload.get("num_samples", "")),
            "max_eval_batches": payload.get("max_eval_batches", ""),
            "batch_size": train_meta.get("batch_size", payload.get("batch_size", "")),
            "num_workers": payload.get("num_workers", ""),
            "eval_subset_id": payload.get("eval_subset_id", ""),
            "gap_over_error": payload.get("gap_over_error", ""),
            "r_lite_scope": payload.get("r_lite_scope", ""),
            "gpu_name": train_meta.get("gpu_name", payload.get("gpu_name", "")),
            "amp": train_meta.get("amp", payload.get("amp", "")),
            "session_id": train_meta.get("session_id", payload.get("session_id", "")),
            "max_wall_seconds": train_meta.get("max_wall_seconds", payload.get("max_wall_seconds", "")),
            "experiment_group": train_meta.get("experiment_group", payload.get("experiment_group", "")),
        }
        try:
            record["gap_over"] = float(record["pgd20_aalite_acc"]) - float(record["r_lite"])
        except (TypeError, ValueError):
            pass
        if isinstance(attacks, dict):
            for attack_name, result in attacks.items():
                key = "pgd20_aalite_acc" if attack_name == "pgd20" else f"{attack_name}_acc"
                if result.get("status") == "ok":
                    record[key] = result.get("robust_acc")
                else:
                    record[key] = ""
                    record[f"{attack_name}_status"] = result.get("status", "")
                    record[f"{attack_name}_error"] = result.get("error", "")
            try:
                record["gap_over"] = float(record["pgd20_aalite_acc"]) - float(record["r_lite"])
            except (TypeError, ValueError):
                pass
        records.append(record)
    return pd.DataFrame.from_records(records)


def _aalite_subset_index(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if not _is_ok(row):
            continue
        try:
            subset_size = int(row.get("subset_size") or row.get("num_samples") or 0)
        except (TypeError, ValueError):
            continue
        indexed[(_source_exp_id(row), subset_size)] = row
    return indexed


def _flatten_aa_subset(
    autoattack_rows: list[dict[str, Any]],
    aalite_subset_rows: list[dict[str, Any]],
    training_metrics: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    lite_by_key = _aalite_subset_index(aalite_subset_rows)
    records: list[dict[str, Any]] = []
    for row in autoattack_rows:
        if not _is_ok(row):
            continue
        exp_id = _source_exp_id(row)
        try:
            subset_size = int(row.get("subset_size") or row.get("num_samples") or 0)
        except (TypeError, ValueError):
            subset_size = 0
        lite_row = lite_by_key.get((exp_id, subset_size))
        r_lite_subset = row.get("r_lite_subset")
        if lite_row is not None:
            r_lite_subset = lite_row.get("r_lite", lite_row.get("r_lite_subset", r_lite_subset))
        aa_subset_acc = row.get("aa_subset_acc")
        bias_value = ""
        dev_value = ""
        if r_lite_subset is not None and aa_subset_acc is not None:
            try:
                bias_value = bias_aa(float(r_lite_subset), float(aa_subset_acc))
                dev_value = dev_aa(float(r_lite_subset), float(aa_subset_acc))
            except (TypeError, ValueError):
                bias_value = ""
                dev_value = ""
        train_meta = (training_metrics or {}).get(str(exp_id), {})
        records.append(
            {
                "exp_id": exp_id,
                "model": row.get("model", ""),
                "defense": train_meta.get("defense", row.get("defense", "")),
                "r_lite_subset": "" if r_lite_subset is None else r_lite_subset,
                "aa_subset_acc": "" if aa_subset_acc is None else aa_subset_acc,
                "bias_aa": bias_value,
                "dev_aa": dev_value,
                "square_acc": row.get("square_acc", ""),
                "subset_size": subset_size or "",
                "status": row.get("status", ""),
                "mode": row.get("mode", ""),
                "dataset_name": row.get("dataset_name", row.get("dataset", "")),
            }
        )
    return pd.DataFrame.from_records(records)


def aggregate(input_dir: Path, output_dir: Path, allow_smoke: bool = False) -> dict[str, str]:
    ensure_dir(output_dir)
    json_rows = _load_json_files(input_dir)
    training_metrics = _load_training_metrics(ROOT / "checkpoints")
    if not allow_smoke:
        _reject_smoke_or_fake(json_rows)
    aalite_rows = _rows_from_subdir_or_filter(
        input_dir,
        "aalite",
        json_rows,
        lambda row: ("r_lite" in row or "attacks" in row) and "aa_subset_acc" not in row,
    )
    standalone_attack_rows = _rows_from_subdir_or_filter(
        input_dir,
        "attacks",
        json_rows,
        lambda row: "attack" in row and "robust_acc" in row and "r_lite" not in row and not isinstance(row.get("attacks"), dict),
    )
    autoattack_subset_rows = _rows_from_subdir_or_filter(input_dir, "autoattack_subset", json_rows, lambda row: "aa_subset_acc" in row)
    aalite_aa_subset_rows = _rows_from_subdir_or_filter(
        input_dir,
        "aalite_aa_subset",
        json_rows,
        lambda row: ("r_lite" in row or "attacks" in row) and "aa_subset_acc" not in row and "aa_subset" in row.get("_source", ""),
    )
    diagnostics_rows = _rows_from_subdir_or_filter(input_dir, "diagnostics", json_rows, lambda row: "diagnosis_pass" in row)
    if not allow_smoke:
        _reject_smoke_or_fake(aalite_rows + standalone_attack_rows + autoattack_subset_rows + aalite_aa_subset_rows + diagnostics_rows)

    outputs: dict[str, str] = {}
    main = _flatten_aalite(aalite_rows, training_metrics, _standalone_attack_index(standalone_attack_rows))
    main_path = output_dir / "main_robustness.csv"
    main.to_csv(main_path, index=False)
    outputs["main_robustness"] = str(main_path)

    aa_df = _flatten_aa_subset(autoattack_subset_rows, aalite_aa_subset_rows, training_metrics)
    if aa_df.empty:
        aa_df = pd.DataFrame(
            columns=[
                "exp_id",
                "model",
                "defense",
                "r_lite_subset",
                "aa_subset_acc",
                "bias_aa",
                "dev_aa",
                "square_acc",
                "subset_size",
                "status",
                "mode",
                "dataset_name",
            ]
        )
    aa_path = output_dir / "aa_subset_check.csv"
    aa_df.to_csv(aa_path, index=False)
    outputs["aa_subset_check"] = str(aa_path)

    diag_records = []
    for row in diagnostics_rows:
        diag_records.append(
            {
                "exp_id": Path(row.get("_source", "diagnostics")).stem,
                "dataset_name": row.get("dataset_name", row.get("dataset", "")),
                "mode": row.get("mode", ""),
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
        invalid_groups: set[str] = set()
        group_key = "experiment_group" if "experiment_group" in main.columns else None
        if group_key:
            for group_name, group in main.groupby(main[group_key].fillna("")):
                if not group_name:
                    continue
                for column in ["gpu_name", "batch_size", "amp", "session_id"]:
                    values = {str(value) for value in group[column].fillna("") if str(value) != ""}
                    if len(values) > 1:
                        invalid_groups.add(str(group_name))
                        break
        for _, row in main.iterrows():
            r_value = row.get("r_lite", "")
            clean_value = row.get("clean_acc", "")
            gpu_hours = row.get("train_time_gpu_hours", "")
            try:
                r_float = float(r_value)
                clean_float = float(clean_value)
                gpu_float = float(gpu_hours)
            except (TypeError, ValueError):
                continue
            if gpu_float <= 0:
                continue
            experiment_group = str(row.get("experiment_group", "") or "")
            budget_rows.append(
                {
                    "experiment_group": experiment_group,
                    "defense": row.get("defense", ""),
                    "epoch_budget": "",
                    "gpu_hour_budget": gpu_float,
                    "gpu_name": row.get("gpu_name", ""),
                    "batch_size": row.get("batch_size", ""),
                    "amp": row.get("amp", ""),
                    "session_id": row.get("session_id", ""),
                    "max_wall_seconds": row.get("max_wall_seconds", ""),
                    "equal_budget_invalid": experiment_group in invalid_groups,
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate raw JSON results into tables.")
    parser.add_argument("--input", default="results/real/raw")
    parser.add_argument("--output", default="results/real/tables")
    parser.add_argument("--allow-smoke", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    outputs = aggregate(ROOT / args.input, ROOT / args.output, allow_smoke=args.allow_smoke)
    print(outputs)


if __name__ == "__main__":
    main()
