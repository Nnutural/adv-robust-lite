from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from _bootstrap import add_project_root

ROOT = add_project_root()

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter


TABLE_DIR = ROOT / "results" / "real" / "tables"
RAW_DIR = ROOT / "results" / "real" / "raw"
CHECKPOINT_DIR = ROOT / "checkpoints"
OUT_DIR = ROOT / "results" / "real" / "figures_nature"

PRIMARY_DEFENSES = ["standard", "fgsm_at", "pgd_at", "fixed_mixed_at"]
AT_DEFENSES = ["fgsm_at", "pgd_at", "fixed_mixed_at"]

DEFENSE_LABELS = {
    "standard": "Standard",
    "fgsm_at": "FGSM-AT",
    "pgd_at": "PGD-AT",
    "fixed_mixed_at": "Fixed mixed AT",
    "trap_logit": "Trap-A logit",
    "trap_random_eot": "Trap-B EOT",
}

DEFENSE_COLORS = {
    "standard": "#8C8C8C",
    "fgsm_at": "#3B73B9",
    "pgd_at": "#7A5AA6",
    "fixed_mixed_at": "#D9822B",
    "trap_logit": "#C9C9C9",
    "trap_random_eot": "#C9C9C9",
}

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#4D4D4D",
        "axes.labelcolor": "#222222",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)
sns.set_theme(
    context="paper",
    style="whitegrid",
    rc={
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "grid.color": "#E6E6E6",
        "grid.linewidth": 0.55,
        "axes.spines.right": False,
        "axes.spines.top": False,
    },
)


def require_file(path: Path) -> Path:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required real result file is missing or empty: {path.relative_to(ROOT)}")
    return path


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(require_file(path))


def pct(value: float) -> float:
    return 100.0 * float(value)


def pct_formatter() -> FuncFormatter:
    return FuncFormatter(lambda x, _: f"{x:.0f}%")


def method_label(defense: str) -> str:
    return DEFENSE_LABELS.get(defense, defense.replace("_", " ").title())


def group_label(group: str) -> str:
    mapping = {
        "g1_structure": "G1",
        "g2_defense": "G2",
        "g5_equal_gpu_hours": "G5",
    }
    return mapping.get(group, str(group))


def save_pub(fig: mpl.figure.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [
        stem.with_suffix(".png"),
        stem.with_suffix(".pdf"),
        stem.with_suffix(".svg"),
    ]
    fig.savefig(outputs[0], dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(outputs[1], bbox_inches="tight", pad_inches=0.05)
    fig.savefig(outputs[2], bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return outputs


def annotate_panel(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        ha="left",
    )


def style_axis(ax: mpl.axes.Axes, *, y_percent: bool = True) -> None:
    ax.grid(True, axis="y", color="#E6E6E6", linewidth=0.55)
    ax.grid(False, axis="x")
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    if y_percent:
        ax.yaxis.set_major_formatter(pct_formatter())


def load_main_rows() -> pd.DataFrame:
    main = read_csv(TABLE_DIR / "main_robustness.csv")
    for col in ["clean_acc", "pgd20_acc", "r_lite", "gap_over", "seed"]:
        if col in main.columns:
            main[col] = pd.to_numeric(main[col], errors="coerce")
    return main


def select_primary_rows(main: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for defense in PRIMARY_DEFENSES:
        candidates = main[
            (main["model"].astype(str) == "preact_resnet18")
            & (main["defense"].astype(str) == defense)
            & (main["seed"] == 0)
            & (main["status"].astype(str) == "ok")
        ].copy()
        if candidates.empty:
            raise ValueError(f"No seed0 preact_resnet18 real row found for defense={defense}")
        if "experiment_group" in candidates.columns and (candidates["experiment_group"] == "g2_defense").any():
            candidates = candidates[candidates["experiment_group"] == "g2_defense"]
        rows.append(candidates.iloc[0])
    primary = pd.DataFrame(rows).reset_index(drop=True)
    primary["defense"] = pd.Categorical(primary["defense"], categories=PRIMARY_DEFENSES, ordered=True)
    return primary.sort_values("defense").reset_index(drop=True)


def trap_rows(main: pd.DataFrame) -> pd.DataFrame:
    traps = main[main["exp_id"].astype(str).str.contains("trap_", na=False)].copy()
    if traps.empty:
        return traps
    traps["trap_kind"] = np.where(
        traps["exp_id"].astype(str).str.contains("trap_logit"),
        "trap_logit",
        "trap_random_eot",
    )
    traps["gap_for_plot"] = traps["pgd20_acc"].astype(float) - traps["r_lite"].astype(float)
    return traps


def make_main_robustness(primary: pd.DataFrame) -> list[Path]:
    metrics = [
        ("clean_acc", "Clean"),
        ("pgd20_acc", "PGD-20"),
        ("r_lite", "AA-Lite R_lite"),
    ]
    fig, ax = plt.subplots(figsize=(7.1, 3.05))
    x = np.arange(len(metrics))
    width = 0.17
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(PRIMARY_DEFENSES))
    for offset, defense in zip(offsets, PRIMARY_DEFENSES, strict=True):
        row = primary[primary["defense"].astype(str) == defense].iloc[0]
        values = [pct(row[col]) for col, _ in metrics]
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=DEFENSE_COLORS[defense],
            edgecolor="white",
            linewidth=0.6,
            label=method_label(defense),
            zorder=3,
        )
        for bar, value in zip(bars, values, strict=True):
            if value < 1.5:
                text_y = 1.8
                label = "0"
            else:
                text_y = value + 1.4
                label = f"{value:.0f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                text_y,
                label,
                ha="center",
                va="bottom",
                fontsize=6,
                color="#222222",
            )

    ax.set_xticks(x, [name for _, name in metrics])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 104)
    ax.set_title("Clean accuracy does not imply adversarial robustness", loc="left", fontsize=8, pad=8)
    style_axis(ax)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.22), handlelength=1.1, columnspacing=1.0)
    ax.annotate(
        "standard robust acc = 0%",
        xy=(x[1] + offsets[0], 2.0),
        xytext=(x[1] - 0.65, 20),
        arrowprops=dict(arrowstyle="-|>", color="#555555", lw=0.7),
        fontsize=7,
        color="#333333",
        ha="left",
    )
    fig.tight_layout()
    return save_pub(fig, OUT_DIR / "figure_main_robustness")


def make_gap_over(primary: pd.DataFrame, main: pd.DataFrame) -> list[Path]:
    honest = primary.copy()
    honest["plot_label"] = honest["defense"].astype(str).map(method_label)
    honest["gap_for_plot"] = honest["pgd20_acc"].astype(float) - honest["r_lite"].astype(float)
    honest["kind"] = "honest"

    traps = trap_rows(main)
    if not traps.empty:
        traps["plot_label"] = traps["trap_kind"].map(DEFENSE_LABELS) + "\nfailed trap\nlimitation"
        traps["kind"] = "trap"
        plot_df = pd.concat(
            [honest[["plot_label", "gap_for_plot", "defense", "kind"]], traps[["plot_label", "gap_for_plot", "trap_kind", "kind"]].rename(columns={"trap_kind": "defense"})],
            ignore_index=True,
        )
    else:
        plot_df = honest[["plot_label", "gap_for_plot", "defense", "kind"]].copy()

    fig, ax = plt.subplots(figsize=(7.1, 2.85))
    x = np.arange(len(plot_df))
    colors = [DEFENSE_COLORS.get(str(row.defense), "#C9C9C9") for row in plot_df.itertuples()]
    hatches = ["///" if row.kind == "trap" else "" for row in plot_df.itertuples()]
    bars = ax.bar(
        x,
        [pct(v) for v in plot_df["gap_for_plot"]],
        color=colors,
        edgecolor="#666666",
        linewidth=0.5,
        zorder=3,
    )
    for bar, hatch, value in zip(bars, hatches, plot_df["gap_for_plot"], strict=True):
        bar.set_hatch(hatch)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            pct(value) + 0.16,
            f"{pct(value):.1f}",
            ha="center",
            va="bottom",
            fontsize=6,
        )

    ax.axhspan(0, 3.0, color="#EFEFEF", alpha=0.5, zorder=0)
    ax.text(0.02, 0.88, "small PGD-20 vs AA-Lite overestimate band", transform=ax.transAxes, fontsize=6, color="#555555")
    ax.set_xticks(x, plot_df["plot_label"], rotation=25, ha="right")
    ax.set_ylabel("PGD-20 - R_lite")
    ax.set_ylim(0, max(4.5, pct(plot_df["gap_for_plot"].max()) + 1.0))
    ax.set_title("Honest adversarial training shows a small Gap_over", loc="left", fontsize=8, pad=8)
    style_axis(ax)
    fig.tight_layout()
    return save_pub(fig, OUT_DIR / "figure_gap_over")


def load_metrics(exp_id: str) -> dict:
    with require_file(CHECKPOINT_DIR / exp_id / "metrics.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_train_log(exp_id: str) -> pd.DataFrame:
    df = read_csv(CHECKPOINT_DIR / exp_id / "train_log.csv")
    required = {"epoch", "val_acc", "pgd7_val_acc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{exp_id}/train_log.csv is missing required columns: {sorted(missing)}")
    return df


def make_training_dynamics() -> list[Path]:
    exp_ids = {
        "fgsm_at": "preact_resnet18_fgsm_at_seed0",
        "pgd_at": "preact_resnet18_pgd_at_seed0",
        "fixed_mixed_at": "preact_resnet18_fixed_mixed_at_seed0",
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), sharey=True)
    for panel, (ax, defense) in enumerate(zip(axes, AT_DEFENSES, strict=True)):
        exp_id = exp_ids[defense]
        log = load_train_log(exp_id)
        metrics = load_metrics(exp_id)
        epochs = pd.to_numeric(log["epoch"], errors="raise")
        clean = 100.0 * pd.to_numeric(log["val_acc"], errors="raise")
        robust = 100.0 * pd.to_numeric(log["pgd7_val_acc"], errors="raise")
        color = DEFENSE_COLORS[defense]

        ax.plot(epochs, clean, color=color, lw=1.35, marker="o", markersize=2.4, label="val acc", zorder=3)
        ax.plot(
            epochs,
            robust,
            color=color,
            lw=1.35,
            ls=(0, (3, 2)),
            marker="s",
            markersize=2.2,
            label="PGD-7 val acc",
            zorder=3,
        )

        best_epoch = int(metrics["best_epoch"])
        best_row = log.loc[log["epoch"] == best_epoch]
        if best_row.empty:
            raise ValueError(f"best_epoch={best_epoch} is not present in {exp_id}/train_log.csv")
        best_robust = pct(float(best_row["pgd7_val_acc"].iloc[0]))
        ax.axvline(best_epoch, color="#333333", lw=0.75, ls=":", zorder=2)
        ax.scatter([best_epoch], [best_robust], color="#222222", s=14, zorder=4)
        ax.text(
            best_epoch + 0.35,
            min(86, best_robust + 8),
            f"best\n{best_epoch}",
            fontsize=6,
            color="#222222",
            ha="left",
            va="center",
        )

        if defense == "fgsm_at" and metrics.get("co_detected") is True:
            co_epoch = int(metrics["co_epoch"])
            co_row = log.loc[log["epoch"] == co_epoch]
            if co_row.empty:
                raise ValueError(f"co_epoch={co_epoch} is not present in {exp_id}/train_log.csv")
            co_robust = pct(float(co_row["pgd7_val_acc"].iloc[0]))
            ax.axvline(co_epoch, color="#555555", lw=0.8, ls=(0, (2, 2)), zorder=2)
            ax.scatter([co_epoch], [co_robust], facecolor="white", edgecolor="#222222", s=22, zorder=5)
            ax.annotate(
                "CO detected\nepoch 32",
                xy=(co_epoch, co_robust),
                xytext=(co_epoch - 8.5, co_robust + 21),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#555555"),
                fontsize=6,
                ha="left",
                va="center",
                color="#222222",
            )

        annotate_panel(ax, chr(ord("a") + panel))
        ax.set_title(method_label(defense), loc="left", fontsize=8, pad=8, color=color)
        ax.set_xlabel("Epoch")
        ax.set_ylim(0, 90)
        ax.set_xlim(max(0.5, epochs.min() - 0.5), epochs.max() + 0.8)
        style_axis(ax)
        if panel == 0:
            ax.set_ylabel("Validation accuracy")
            ax.legend(loc="lower left", fontsize=6, handlelength=1.8)

    fig.suptitle("Training dynamics expose the FGSM-AT robust-validation collapse", x=0.02, y=1.04, ha="left", fontsize=8)
    fig.tight_layout()
    return save_pub(fig, OUT_DIR / "figure_training_dynamics")


def attach_seed_labels(budget: pd.DataFrame, main: pd.DataFrame) -> pd.DataFrame:
    rows = []
    main_valid = main.copy()
    main_valid["defense"] = main_valid["defense"].astype(str)
    for _, row in budget.iterrows():
        candidates = main_valid[
            (main_valid["experiment_group"].astype(str) == str(row["experiment_group"]))
            & (main_valid["defense"].astype(str) == str(row["defense"]))
            & (main_valid["status"].astype(str) == "ok")
        ].copy()
        if candidates.empty:
            seed = "?"
            exp_id = ""
        else:
            candidates["hour_delta"] = (candidates["train_time_gpu_hours"].astype(float) - float(row["gpu_hour_budget"])).abs()
            match = candidates.sort_values("hour_delta").iloc[0]
            seed = int(match["seed"])
            exp_id = str(match["exp_id"])
        enriched = row.to_dict()
        enriched["seed"] = seed
        enriched["exp_id"] = exp_id
        rows.append(enriched)
    return pd.DataFrame(rows)


def make_cost_tradeoff(main: pd.DataFrame) -> list[Path]:
    budget = read_csv(TABLE_DIR / "budget_comparison.csv")
    for col in ["gpu_hour_budget", "clean_acc", "r_lite"]:
        budget[col] = pd.to_numeric(budget[col], errors="coerce")
    plot_df = budget[budget["gpu_hour_budget"] > 0].copy()
    if plot_df.empty:
        raise ValueError("budget_comparison.csv has no rows with gpu_hour_budget > 0")
    present_groups = set(plot_df["experiment_group"].astype(str))
    if not {"g2_defense", "g5_equal_gpu_hours"}.issubset(present_groups):
        raise ValueError("figure_cost_tradeoff requires both G2 and G5 budget rows")
    plot_df = attach_seed_labels(plot_df, main)

    fig, ax = plt.subplots(figsize=(6.0, 3.55))
    cmap = mpl.colormaps["cividis"]
    norm = mpl.colors.Normalize(vmin=pct(plot_df["clean_acc"].min()), vmax=pct(plot_df["clean_acc"].max()))
    marker_map = {"G1": "s", "G2": "o", "G5": "^"}
    label_offsets = {
        ("G1", "standard", 0): (0.035, 1.1, "left"),
        ("G2", "standard", 0): (-0.035, 1.5, "right"),
        ("G2", "fgsm_at", 0): (-0.035, 2.8, "right"),
        ("G2", "pgd_at", 0): (-0.040, 1.1, "right"),
        ("G2", "fixed_mixed_at", 0): (0.035, 1.9, "left"),
        ("G5", "fgsm_at", 1): (0.035, 2.8, "left"),
        ("G5", "pgd_at", 1): (0.035, 1.5, "left"),
        ("G5", "fixed_mixed_at", 1): (-0.040, 1.9, "right"),
    }
    short_labels = {
        "standard": "Std",
        "fgsm_at": "FGSM",
        "pgd_at": "PGD",
        "fixed_mixed_at": "Mix",
    }
    for _, row in plot_df.iterrows():
        g = group_label(str(row["experiment_group"]))
        defense = str(row["defense"])
        marker = marker_map.get(g, "o")
        x = float(row["gpu_hour_budget"])
        y = pct(row["r_lite"])
        clean_pct = pct(row["clean_acc"])
        ax.scatter(
            x,
            y,
            s=62,
            marker=marker,
            color=cmap(norm(clean_pct)),
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        seed = int(row["seed"]) if str(row["seed"]).isdigit() else row["seed"]
        label = f"{short_labels.get(defense, defense)} s{seed}"
        dx, dy, ha = label_offsets.get((g, defense, seed), (0.035, 1.2, "left"))
        ax.text(x + dx, y + dy, label, fontsize=5.8, ha=ha, va="bottom", color="#222222")

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.055)
    cbar.set_label("Clean acc")
    cbar.ax.yaxis.set_major_formatter(pct_formatter())

    handles = [
        mpl.lines.Line2D([0], [0], marker=marker, color="none", markerfacecolor="#888888", markeredgecolor="white", markersize=6, label=group)
        for group, marker in marker_map.items()
        if group in {group_label(str(g)) for g in plot_df["experiment_group"]}
    ]
    ax.legend(handles=handles, loc="upper left", title="Group", fontsize=6, title_fontsize=6)
    ax.set_xlabel("Training GPU hours")
    ax.set_ylabel("AA-Lite R_lite")
    ax.set_title("Robustness improves with adversarial training budget, but clean accuracy trades off", loc="left", fontsize=8, pad=8)
    ax.set_ylim(-2, max(48, pct(plot_df["r_lite"].max()) + 7))
    ax.set_xlim(max(0, plot_df["gpu_hour_budget"].min() - 0.12), plot_df["gpu_hour_budget"].max() + 0.15)
    style_axis(ax)
    fig.tight_layout()
    return save_pub(fig, OUT_DIR / "figure_cost_tradeoff")


def parse_defense_from_exp_id(exp_id: str) -> str:
    for defense in ["fixed_mixed_at", "fgsm_at", "pgd_at", "standard"]:
        if defense in exp_id:
            return defense
    raise ValueError(f"Cannot infer defense from exp_id={exp_id}")


def diagnostics_payloads() -> list[dict]:
    files = sorted((RAW_DIR / "diagnostics").glob("*.json"))
    if not files:
        raise FileNotFoundError("No diagnostics JSON files found under results/real/raw/diagnostics")
    payloads = []
    for path in files:
        with require_file(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("diagnosis_pass") is not True:
            raise ValueError(f"Diagnostics did not pass for {path.name}")
        payloads.append(payload)
    order = {defense: idx for idx, defense in enumerate(PRIMARY_DEFENSES)}
    return sorted(payloads, key=lambda item: order[parse_defense_from_exp_id(str(item["exp_id"]))])


def make_diagnostics() -> list[Path]:
    payloads = diagnostics_payloads()
    scan_specs = [
        ("eps_scan", "epsilon", "Epsilon (x / 255)", lambda row: float(row["eps"]) * 255.0),
        ("step_scan", "steps", "PGD steps", lambda row: float(row["steps"])),
        ("restart_scan", "restarts", "Restarts", lambda row: float(row["restarts"])),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.85), sharey=True)
    for panel, (ax, (scan_key, _, xlabel, x_fn)) in enumerate(zip(axes, scan_specs, strict=True)):
        for payload in payloads:
            defense = parse_defense_from_exp_id(str(payload["exp_id"]))
            rows = payload.get(scan_key, [])
            if not rows:
                raise ValueError(f"{payload['exp_id']} is missing {scan_key}")
            x = [x_fn(row) for row in rows]
            y = [pct(float(row["robust_acc"])) for row in rows]
            ax.plot(
                x,
                y,
                marker="o",
                markersize=2.8,
                lw=1.25,
                color=DEFENSE_COLORS[defense],
                label=method_label(defense),
                zorder=3,
            )
            if panel == 0:
                ax.text(x[-1] + 0.4, y[-1], method_label(defense), fontsize=6, va="center", color=DEFENSE_COLORS[defense])
        annotate_panel(ax, chr(ord("a") + panel))
        ax.set_xlabel(xlabel)
        ax.set_title(["Epsilon scan", "Step scan", "Restart scan"][panel], loc="left", fontsize=8, pad=8)
        style_axis(ax)
        if scan_key == "eps_scan":
            ax.set_xticks([2, 4, 8, 16])
            ax.set_xlim(1.2, 18.5)
        elif scan_key == "step_scan":
            ax.set_xticks([10, 20, 50])
            ax.set_xlim(7, 53)
        else:
            ax.set_xticks([1, 3, 5])
            ax.set_xlim(0.5, 5.5)
        if panel == 0:
            ax.set_ylabel("Robust accuracy")

    axes[0].set_ylim(-2, 76)
    axes[2].text(
        0.96,
        0.92,
        "all pass monotonicity\nand restart sanity",
        transform=axes[2].transAxes,
        ha="right",
        va="top",
        fontsize=6,
        color="#333333",
    )
    fig.suptitle("Attack-strength scans support the main models' sanity checks", x=0.02, y=1.04, ha="left", fontsize=8)
    fig.tight_layout()
    return save_pub(fig, OUT_DIR / "figure_diagnostics")


def per_class_file(defense: str) -> Path:
    return RAW_DIR / "attacks" / f"preact_resnet18_{defense}_seed0_pgd20_per_class.csv"


def make_per_class_appendix() -> list[Path]:
    rows = []
    for defense in AT_DEFENSES:
        df = read_csv(per_class_file(defense))
        required = {"class", "num_samples", "robust_acc"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{per_class_file(defense).name} is missing required columns: {sorted(missing)}")
        df["defense"] = defense
        rows.append(df)
    plot_df = pd.concat(rows, ignore_index=True)
    plot_df["class"] = pd.to_numeric(plot_df["class"], errors="raise").astype(int)
    plot_df["robust_acc"] = pd.to_numeric(plot_df["robust_acc"], errors="raise")

    fig, ax = plt.subplots(figsize=(7.2, 3.15))
    for defense in AT_DEFENSES:
        sub = plot_df[plot_df["defense"] == defense].sort_values("class")
        x = sub["class"].to_numpy()
        y = 100.0 * sub["robust_acc"].to_numpy()
        ax.plot(
            x,
            y,
            marker="o",
            markersize=3.2,
            lw=1.35,
            color=DEFENSE_COLORS[defense],
            label=method_label(defense),
            zorder=3,
        )
        ax.text(x[-1] + 0.16, y[-1], method_label(defense), fontsize=6, va="center", color=DEFENSE_COLORS[defense])

    ax.set_xticks(range(10), CIFAR10_CLASSES, rotation=28, ha="right")
    ax.set_xlim(-0.35, 10.15)
    ax.set_ylim(0, max(75, 100.0 * plot_df["robust_acc"].max() + 6))
    ax.set_xlabel("CIFAR-10 class")
    ax.set_ylabel("PGD-20 robust accuracy")
    ax.set_title("Class-level PGD-20 robustness remains uneven across adversarial-training recipes", loc="left", fontsize=8, pad=8)
    style_axis(ax)
    fig.tight_layout()
    return save_pub(fig, OUT_DIR / "figure_per_class_appendix")


def verify_outputs(outputs: Iterable[Path]) -> None:
    missing = [path for path in outputs if not path.exists() or path.stat().st_size == 0]
    if missing:
        names = ", ".join(str(path.relative_to(ROOT)) for path in missing)
        raise RuntimeError(f"Empty or missing figure outputs: {names}")


def write_manifest(output_map: dict[str, list[Path]]) -> Path:
    conclusions = {
        "figure_main_robustness": "Standard training reaches the highest clean accuracy but has 0% PGD-20 and AA-Lite robustness, whereas adversarial training provides real robust accuracy.",
        "figure_gap_over": "For honest adversarial training, PGD-20 only slightly overestimates AA-Lite R_lite; trap rows, where present, are marked only as failed-trap limitations.",
        "figure_training_dynamics": "FGSM-AT seed0 shows catastrophic robust-validation collapse at epoch 32, while PGD-AT and fixed mixed AT retain their best robust-validation checkpoints.",
        "figure_cost_tradeoff": "Positive GPU-hour adversarial-training runs improve AA-Lite robustness, with G2 and G5 points showing the cost and clean-accuracy trade-off.",
        "figure_diagnostics": "Real diagnostics pass epsilon, step, and restart sanity scans, supporting the main robustness measurements against gradient-masking concerns.",
        "figure_per_class_appendix": "PGD-20 robustness varies strongly by CIFAR-10 class, with no adversarial-training recipe uniformly dominating every class.",
    }
    source_notes = {
        "figure_main_robustness": "results/real/tables/main_robustness.csv",
        "figure_gap_over": "results/real/tables/main_robustness.csv",
        "figure_training_dynamics": "checkpoints/*/train_log.csv and checkpoints/*/metrics.json",
        "figure_cost_tradeoff": "results/real/tables/budget_comparison.csv plus seed labels from results/real/tables/main_robustness.csv",
        "figure_diagnostics": "results/real/raw/diagnostics/*.json",
        "figure_per_class_appendix": "results/real/raw/attacks/*_pgd20_per_class.csv",
    }

    lines = [
        "# Nature-style Figure Manifest",
        "",
        "Backend: Python only (matplotlib/seaborn/pandas).",
        "Archetype: quantitative grid with validation and appendix panels.",
        "Export contract: PNG at 600 dpi plus editable PDF and SVG for every figure.",
        "Data policy: existing real-result files only; no simulated data and no experiment reruns.",
        "",
    ]
    for figure_name, paths in output_map.items():
        rel_paths = ", ".join(str(path.relative_to(ROOT)).replace("\\", "/") for path in paths)
        lines.extend(
            [
                f"## {figure_name}",
                f"- Core conclusion: {conclusions[figure_name]}",
                f"- Source data: {source_notes[figure_name]}",
                f"- Outputs: {rel_paths}",
                "",
            ]
        )
    manifest = OUT_DIR / "figure_manifest.md"
    manifest.write_text("\n".join(lines), encoding="utf-8")
    return manifest


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main_df = load_main_rows()
    primary = select_primary_rows(main_df)

    output_map = {
        "figure_main_robustness": make_main_robustness(primary),
        "figure_gap_over": make_gap_over(primary, main_df),
        "figure_training_dynamics": make_training_dynamics(),
        "figure_cost_tradeoff": make_cost_tradeoff(main_df),
        "figure_diagnostics": make_diagnostics(),
        "figure_per_class_appendix": make_per_class_appendix(),
    }
    all_outputs = [path for paths in output_map.values() for path in paths]
    verify_outputs(all_outputs)
    manifest = write_manifest(output_map)
    verify_outputs([manifest])

    print("Generated Nature-style figures:")
    for figure_name, paths in output_map.items():
        print(f"- {figure_name}: " + ", ".join(str(path.relative_to(ROOT)) for path in paths))
    print(f"- manifest: {manifest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
