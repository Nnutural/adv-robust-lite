from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_project_root

ROOT = add_project_root()

import pandas as pd
import numpy as np

from src.visualization.curves import save_line_curve
from src.visualization.heatmap import save_heatmap
from src.visualization.tradeoff import save_tradeoff


ATTACK_COLUMNS = ["fgsm_acc", "pgd20_acc", "apgd_ce_acc", "apgd_dlr_acc"]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def make_figures(tables_dir: Path, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    main = _read_csv(tables_dir / "main_robustness.csv")
    if not main.empty:
        available = [col for col in ATTACK_COLUMNS if col in main.columns]
        matrix = main[available].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float) if available else [[np.nan]]
        row_labels = (main.get("exp_id", pd.Series([f"run{i}" for i in range(len(main))])).astype(str)).tolist()
        col_labels = available or ["no_data"]
    else:
        matrix = [[0.0]]
        row_labels = ["no results"]
        col_labels = ["robust acc"]
    base = output_dir / "fig2_structure_heatmap"
    save_heatmap(matrix, row_labels, col_labels, "Robustness heatmap", base)
    outputs["fig2_structure_heatmap"] = str(base.with_suffix(".png"))

    if not main.empty and {"pgd20_acc", "r_lite"}.issubset(main.columns):
        gap_base = output_dir / "fig4_gap_over"
        labels = main.get("exp_id", pd.Series(range(len(main)))).astype(str).tolist()
        pgd = main["pgd20_acc"].fillna(0.0).astype(float).tolist()
        lite = main["r_lite"].fillna(0.0).astype(float).tolist()
        import matplotlib.pyplot as plt
        x = np.arange(len(labels))
        width = 0.36
        fig, ax = plt.subplots(figsize=(max(5.5, len(labels) * 0.8), 3.8))
        ax.bar(x - width / 2, pgd, width, label="PGD-20")
        ax.bar(x + width / 2, lite, width, label="AA-Lite")
        ax.set_xticks(x, labels=labels, rotation=30, ha="right")
        ax.set_ylabel("Robust accuracy")
        ax.set_title("PGD-only vs AA-Lite")
        ax.legend()
        ax.grid(True, axis="y", linewidth=0.5, alpha=0.4)
        fig.tight_layout()
        fig.savefig(gap_base.with_suffix(".png"), dpi=200)
        fig.savefig(gap_base.with_suffix(".pdf"))
        plt.close(fig)
        outputs["fig4_gap_over"] = str(gap_base.with_suffix(".png"))

    trade_base = output_dir / "fig5_cost_robust_tradeoff"
    save_tradeoff(main.to_dict("records") if not main.empty else [], trade_base)
    outputs["fig5_cost_robust_tradeoff"] = str(trade_base.with_suffix(".png"))

    diagnostics = _read_csv(tables_dir / "gradient_masking_diagnostics.csv")
    if not diagnostics.empty:
        raw_diag = ROOT / "results/raw/diagnostics"
        json_files = list(raw_diag.glob("*.json"))
        if json_files:
            import json

            payload = json.loads(json_files[0].read_text(encoding="utf-8"))
            eps_rows = payload.get("eps_scan", [])
            if eps_rows:
                eps = [row["eps"] for row in eps_rows]
                robust = [row.get("robust_acc") or 0.0 for row in eps_rows]
                curve_base = output_dir / "fig6_epsilon_curve"
                save_line_curve(eps, robust, "Epsilon monotonicity", "epsilon", "Robust accuracy", curve_base)
                outputs["fig6_epsilon_curve"] = str(curve_base.with_suffix(".png"))
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-style figures from CSV tables.")
    parser.add_argument("--tables", default="results/tables")
    parser.add_argument("--output", default="results/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = make_figures(ROOT / args.tables, ROOT / args.output)
    print(outputs)


if __name__ == "__main__":
    main()
