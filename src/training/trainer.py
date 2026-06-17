from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.attacks.factory import build_attack_config, AttackFactory
from src.evaluation.metrics import accuracy_from_logits
from src.utils.checkpoint import save_checkpoint
from src.utils.io import ensure_dir, save_json, write_csv
from src.utils.timer import gpu_hours_from_seconds


@dataclass
class TrainConfig:
    model_name: str
    defense: str = "standard"
    epochs: int = 1
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    amp: bool = False
    eps: float = 8 / 255
    pgd_alpha: float = 2 / 255
    pgd_steps: int = 7
    seed: int = 0
    output_dir: str = "checkpoints"
    run_name: str | None = None
    apgd_every_n_batches: int = 5
    max_train_batches: int = 0
    max_eval_batches: int = 0


class Trainer:
    def __init__(self, model: nn.Module, train_loader, val_loader, device: torch.device | str, config: TrainConfig):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device)
        self.config = config
        self.optimizer = SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(1, config.epochs))
        self.scaler = GradScaler(enabled=config.amp and self.device.type == "cuda")
        self.run_name = config.run_name or f"{config.model_name}_{config.defense}_seed{config.seed}"
        self.run_dir = ensure_dir(Path(config.output_dir) / self.run_name)

    def _attack_batch(self, images: torch.Tensor, labels: torch.Tensor, attack_name: str) -> torch.Tensor:
        if attack_name == "clean":
            return images
        if attack_name == "fgsm":
            config = build_attack_config("fgsm", eps=self.config.eps)
        elif attack_name == "pgd":
            config = build_attack_config(
                "pgd",
                eps=self.config.eps,
                steps=self.config.pgd_steps,
                alpha=self.config.pgd_alpha,
            )
        elif attack_name == "apgd_ce":
            config = build_attack_config("apgd_ce", eps=self.config.eps, steps=max(5, self.config.pgd_steps))
        else:
            raise ValueError(f"Unsupported training attack: {attack_name}")
        attack = AttackFactory.create(config)
        return attack(self.model, images, labels)

    def _fixed_mixed_stage(self, epoch_zero_based: int) -> dict[str, float]:
        total = max(1, self.config.epochs)
        progress = epoch_zero_based / total
        if progress < 0.3:
            return {"clean": 0.5, "fgsm": 0.5}
        if progress < 0.75:
            return {"clean": 0.3, "pgd": 0.7}
        return {"clean": 0.3, "pgd": 0.5, "apgd_ce": 0.2}

    def _mix_for_epoch(self, epoch_zero_based: int) -> dict[str, float]:
        defense = self.config.defense.lower()
        if defense == "standard":
            return {"clean": 1.0}
        if defense == "fgsm_at":
            return {"fgsm": 1.0}
        if defense == "pgd_at":
            return {"pgd": 1.0}
        if defense == "fixed_mixed_at":
            return self._fixed_mixed_stage(epoch_zero_based)
        if defense == "budget_scheduler_at":
            return self._fixed_mixed_stage(epoch_zero_based)
        raise ValueError(f"Unsupported defense: {self.config.defense}")

    @staticmethod
    def _choose_attack(mix: dict[str, float]) -> str:
        attacks = list(mix)
        weights = [mix[name] for name in attacks]
        return random.choices(attacks, weights=weights, k=1)[0]

    def train_one_epoch(self, epoch_zero_based: int) -> dict[str, Any]:
        self.model.train()
        total = 0
        total_loss = 0.0
        total_acc = 0.0
        mix = self._mix_for_epoch(epoch_zero_based)
        attack_counts = {key: 0 for key in mix}
        progress = tqdm(self.train_loader, desc=f"train/{self.config.defense}", leave=False)

        for batch_idx, (images, labels) in enumerate(progress):
            if self.config.max_train_batches and batch_idx >= self.config.max_train_batches:
                break
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            attack_name = self._choose_attack(mix)
            if attack_name == "apgd_ce" and batch_idx % max(1, self.config.apgd_every_n_batches) != 0:
                attack_name = "pgd"
            attack_counts[attack_name] = attack_counts.get(attack_name, 0) + 1
            train_images = self._attack_batch(images, labels, attack_name)

            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.config.amp and self.device.type == "cuda"):
                logits = self.model(train_images)
                loss = F.cross_entropy(logits, labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = labels.size(0)
            total += batch_size
            total_loss += loss.item() * batch_size
            total_acc += accuracy_from_logits(logits.detach(), labels) * batch_size
            progress.set_postfix(loss=f"{total_loss / max(1, total):.4f}", acc=f"{total_acc / max(1, total):.4f}")

        return {
            "train_loss": total_loss / max(1, total),
            "train_acc": total_acc / max(1, total),
            "attack_mix": json.dumps(mix, sort_keys=True),
            "attack_counts": json.dumps(attack_counts, sort_keys=True),
        }

    @torch.no_grad()
    def evaluate_clean(self) -> dict[str, float]:
        self.model.eval()
        total = 0
        total_loss = 0.0
        total_acc = 0.0
        for batch_idx, (images, labels) in enumerate(tqdm(self.val_loader, desc="val/clean", leave=False)):
            if self.config.max_eval_batches and batch_idx >= self.config.max_eval_batches:
                break
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            logits = self.model(images)
            loss = F.cross_entropy(logits, labels)
            batch_size = labels.size(0)
            total += batch_size
            total_loss += loss.item() * batch_size
            total_acc += accuracy_from_logits(logits, labels) * batch_size
        return {"val_loss": total_loss / max(1, total), "val_acc": total_acc / max(1, total)}

    def train(self) -> dict[str, Any]:
        start = time.perf_counter()
        history: list[dict[str, Any]] = []
        best_val_acc = -1.0
        best_path = self.run_dir / "best.pt"
        for epoch in range(self.config.epochs):
            train_stats = self.train_one_epoch(epoch)
            val_stats = self.evaluate_clean()
            self.scheduler.step()
            elapsed = time.perf_counter() - start
            row = {
                "epoch": epoch + 1,
                "defense": self.config.defense,
                "lr": self.optimizer.param_groups[0]["lr"],
                "wall_time_sec": elapsed,
                **train_stats,
                **val_stats,
            }
            history.append(row)
            write_csv(self.run_dir / "train_log.csv", history)

            checkpoint = {
                "epoch": epoch + 1,
                "model_name": self.config.model_name,
                "defense": self.config.defense,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "config": self.config.__dict__,
                "history": history,
            }
            save_checkpoint(self.run_dir / "last.pt", checkpoint)
            if val_stats["val_acc"] > best_val_acc:
                best_val_acc = val_stats["val_acc"]
                save_checkpoint(best_path, checkpoint)

        elapsed = time.perf_counter() - start
        device_count = torch.cuda.device_count() if self.device.type == "cuda" else 1
        metrics = {
            "exp_id": self.run_name,
            "model": self.config.model_name,
            "defense": self.config.defense,
            "seed": self.config.seed,
            "clean_acc": best_val_acc,
            "train_time_sec": elapsed,
            "train_time_gpu_hours": gpu_hours_from_seconds(elapsed, device_count=device_count),
            "params": sum(parameter.numel() for parameter in self.model.parameters()),
            "checkpoint_path": str(best_path),
            "device": str(self.device),
            "max_train_batches": self.config.max_train_batches,
            "max_eval_batches": self.config.max_eval_batches,
        }
        if self.device.type == "cuda":
            metrics["gpu_name"] = torch.cuda.get_device_name(0)
        save_json(self.run_dir / "metrics.json", metrics)
        return metrics
