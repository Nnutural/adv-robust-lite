from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from src.attacks.factory import AttackConfig, AttackFactory, AttackUnavailableError
from src.utils.io import save_json, write_csv


class AttackRunner:
    def __init__(
        self,
        attack_config: AttackConfig,
        device: torch.device | str,
        max_samples: int = 0,
        max_eval_batches: int = 0,
        save_samples: int = 0,
    ) -> None:
        self.attack_config = attack_config
        self.attack = AttackFactory.create(attack_config)
        self.device = torch.device(device)
        self.max_samples = max_samples
        self.max_eval_batches = max_eval_batches
        self.save_samples = save_samples

    def run(
        self,
        model: nn.Module,
        dataloader,
        output_json: str | Path | None = None,
        per_class_csv: str | Path | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        model.to(self.device)
        model.eval()
        start = time.perf_counter()
        total = 0
        clean_correct = 0
        adv_correct = 0
        clean_correct_for_success = 0
        attack_success = 0
        confidence_drop = 0.0
        linf_sum = 0.0
        per_class_total = [0] * 10
        per_class_adv_correct = [0] * 10
        sample_payload: dict[str, torch.Tensor] = {}

        try:
            iterator = tqdm(dataloader, desc=f"attack/{self.attack_config.name}", leave=False)
            for batch_idx, (images, labels) in enumerate(iterator):
                if self.max_eval_batches and batch_idx >= self.max_eval_batches:
                    break
                if self.max_samples and total >= self.max_samples:
                    break
                if self.max_samples:
                    remaining = self.max_samples - total
                    images = images[:remaining]
                    labels = labels[:remaining]

                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                with torch.no_grad():
                    clean_logits = model(images)
                    clean_probs = F.softmax(clean_logits, dim=1)
                    clean_preds = clean_logits.argmax(dim=1)
                    clean_true_conf = clean_probs.gather(1, labels.view(-1, 1)).squeeze(1)

                adv_images = self.attack(model, images, labels)
                if adv_images.shape != images.shape:
                    raise RuntimeError(f"Attack returned shape {tuple(adv_images.shape)}, expected {tuple(images.shape)}")
                if adv_images.min().item() < -1e-6 or adv_images.max().item() > 1.0 + 1e-6:
                    raise RuntimeError("Attack returned images outside [0,1]")

                with torch.no_grad():
                    adv_logits = model(adv_images)
                    adv_probs = F.softmax(adv_logits, dim=1)
                    adv_preds = adv_logits.argmax(dim=1)
                    adv_true_conf = adv_probs.gather(1, labels.view(-1, 1)).squeeze(1)

                batch_size = labels.size(0)
                clean_mask = clean_preds == labels
                adv_mask = adv_preds == labels
                clean_correct += clean_mask.sum().item()
                adv_correct += adv_mask.sum().item()
                clean_correct_for_success += clean_mask.sum().item()
                attack_success += (clean_mask & ~adv_mask).sum().item()
                confidence_drop += (clean_true_conf - adv_true_conf).sum().item()
                linf_sum += (adv_images - images).flatten(1).abs().amax(dim=1).sum().item()

                for label, correct in zip(labels.detach().cpu().tolist(), adv_mask.detach().cpu().tolist()):
                    per_class_total[int(label)] += 1
                    per_class_adv_correct[int(label)] += int(correct)

                if self.save_samples and not sample_payload:
                    keep = min(self.save_samples, batch_size)
                    sample_payload = {
                        "images": images[:keep].detach().cpu(),
                        "adv_images": adv_images[:keep].detach().cpu(),
                        "labels": labels[:keep].detach().cpu(),
                    }

                total += batch_size
                iterator.set_postfix(robust_acc=f"{adv_correct / max(1, total):.4f}")

            runtime_sec = time.perf_counter() - start
            result = {
                **asdict(self.attack_config),
                "attack": self.attack_config.name,
                "robust_acc": adv_correct / max(1, total),
                "clean_acc": clean_correct / max(1, total),
                "attack_success_rate": attack_success / max(1, clean_correct_for_success),
                "num_samples": total,
                "runtime_sec": runtime_sec,
                "avg_confidence_drop": confidence_drop / max(1, total),
                "avg_linf": linf_sum / max(1, total),
                "status": "ok",
                "error": None,
                "max_samples": self.max_samples,
                "max_eval_batches": self.max_eval_batches,
            }
        except AttackUnavailableError as exc:
            result = {
                **asdict(self.attack_config),
                "attack": self.attack_config.name,
                "robust_acc": None,
                "attack_success_rate": None,
                "num_samples": total,
                "runtime_sec": time.perf_counter() - start,
                "status": "skipped",
                "error": str(exc),
                "max_samples": self.max_samples,
                "max_eval_batches": self.max_eval_batches,
            }
        except Exception as exc:
            result = {
                **asdict(self.attack_config),
                "attack": self.attack_config.name,
                "robust_acc": None,
                "attack_success_rate": None,
                "num_samples": total,
                "runtime_sec": time.perf_counter() - start,
                "status": "failed",
                "error": str(exc),
                "max_samples": self.max_samples,
                "max_eval_batches": self.max_eval_batches,
            }

        if metadata:
            result.update(dict(metadata))

        if output_json:
            if sample_payload:
                sample_path = Path(output_json).with_name(Path(output_json).stem + "_samples.pt")
                sample_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(sample_payload, sample_path)
                result["samples_path"] = str(sample_path)
            save_json(output_json, result)

        if per_class_csv and result["status"] == "ok":
            rows = []
            for class_idx, count in enumerate(per_class_total):
                rows.append(
                    {
                        "class": class_idx,
                        "num_samples": count,
                        "robust_acc": per_class_adv_correct[class_idx] / max(1, count),
                    }
                )
            write_csv(per_class_csv, rows)

        return result
