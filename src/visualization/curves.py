from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_line_curve(x_values, y_values, title: str, xlabel: str, ylabel: str, output_base: str | Path) -> None:
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(x_values, y_values, marker="o", linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=200)
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)

