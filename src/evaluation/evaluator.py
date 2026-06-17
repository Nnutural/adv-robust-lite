from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.evaluation.metrics import accuracy_from_logits


@torch.no_grad()
def evaluate_clean(
    model,
    dataloader,
    device: torch.device | str,
    max_samples: int = 0,
    max_batches: int = 0,
    max_eval_batches: int = 0,
) -> dict[str, float]:
    device = torch.device(device)
    model.to(device)
    model.eval()
    start = time.perf_counter()
    total = 0
    total_loss = 0.0
    total_acc = 0.0
    batch_limit = max_eval_batches or max_batches
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="eval/clean", leave=False)):
        if batch_limit and batch_idx >= batch_limit:
            break
        if max_samples and total >= max_samples:
            break
        if max_samples:
            remaining = max_samples - total
            images = images[:remaining]
            labels = labels[:remaining]
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        batch_size = labels.size(0)
        total += batch_size
        total_loss += loss.item() * batch_size
        total_acc += accuracy_from_logits(logits, labels) * batch_size
    return {
        "clean_loss": total_loss / max(1, total),
        "clean_acc": total_acc / max(1, total),
        "num_samples": total,
        "runtime_sec": time.perf_counter() - start,
    }
