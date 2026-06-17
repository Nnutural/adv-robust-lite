from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_tradeoff(rows: list[dict], output_base: str | Path) -> None:
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    if rows:
        x = [float(row.get("train_time_gpu_hours", row.get("gpu_hour_budget", 0.0)) or 0.0) for row in rows]
        y = [float(row.get("r_lite", 0.0) or 0.0) for row in rows]
        colors = [float(row.get("clean_acc", 0.0) or 0.0) for row in rows]
        labels = [str(row.get("defense", row.get("exp_id", ""))) for row in rows]
        sc = ax.scatter(x, y, c=colors, cmap="plasma", s=70, edgecolor="black", linewidth=0.5)
        for xi, yi, label in zip(x, y, labels):
            ax.text(xi, yi, label, fontsize=8, ha="left", va="bottom")
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Clean accuracy")
    else:
        ax.text(0.5, 0.5, "No results yet", transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel("Training GPU hours")
    ax.set_ylabel("AA-Lite R_lite")
    ax.set_title("Cost and robustness trade-off")
    ax.grid(True, linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=200)
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)

