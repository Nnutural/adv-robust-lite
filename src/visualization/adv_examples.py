from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_adv_examples(sample_path: str | Path, output_base: str | Path, max_images: int = 8) -> None:
    import torch

    payload = torch.load(sample_path, map_location="cpu")
    images = payload["images"][:max_images]
    adv_images = payload["adv_images"][:max_images]
    labels = payload.get("labels")
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, len(images), figsize=(len(images) * 1.4, 3.0))
    for idx in range(len(images)):
        for row, tensor, prefix in ((0, images[idx], "clean"), (1, adv_images[idx], "adv")):
            ax = axes[row, idx] if len(images) > 1 else axes[row]
            ax.imshow(tensor.permute(1, 2, 0).clamp(0, 1))
            title = prefix if labels is None else f"{prefix}\n{int(labels[idx])}"
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=200)
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)

