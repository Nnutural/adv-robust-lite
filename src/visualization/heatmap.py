from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_heatmap(matrix, row_labels, col_labels, title: str, output_base: str | Path) -> None:
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(matrix, dtype=float)
    masked = np.ma.masked_invalid(data)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#d9d9d9")
    fig, ax = plt.subplots(figsize=(max(5, len(col_labels) * 1.2), max(3.5, len(row_labels) * 0.6)))
    image = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(col_labels)), labels=col_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)), labels=row_labels)
    ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Robust accuracy")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            label = "NA" if np.isnan(value) else f"{value:.2f}"
            color = "black" if np.isnan(value) or value >= 0.5 else "white"
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=200)
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)
