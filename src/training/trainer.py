from __future__ import annotations

import json
import random
import time
import uuid
from dataclasses import dataclass
from itertools import islice
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
from src.utils.timer import format_seconds, gpu_hours_from_seconds


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
    best_criterion: str = "auto"
    robust_val_steps: int = 7
    robust_val_subset_size: int = 0
    eval_every: int = 1
    train_attack: dict[str, Any] | None = None
    eval_attacks: list[str] | None = None
    co_check_enabled: bool = False
    co_check_every: int = 1
    co_threshold: float = 0.15
    batch_size: int = 0
    max_wall_seconds: int = 0
    session_id: str = ""
    experiment_group: str = ""
    dataset_name: str = "cifar10"
    mode: str = "real"
    show_progress: bool = True
    progress_log_every: int = 1


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
        self.session_id = config.session_id or uuid.uuid4().hex[:12]
        self._train_start_time = 0.0

    @staticmethod
    def _loader_total(loader, max_batches: int = 0) -> int | None:
        try:
            total = len(loader)
        except TypeError:
            return max_batches or None
        return min(total, max_batches) if max_batches else total

    @staticmethod
    def _limited_loader(loader, max_batches: int = 0):
        return islice(loader, max_batches) if max_batches else loader

    @staticmethod
    def _fmt(value: Any) -> str:
        if value is None or value == "":
            return "n/a"
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return str(value)

    def _log(self, message: str) -> None:
        if self.config.show_progress:
            tqdm.write(message)

    def _wall_clock_expired(self) -> bool:
        return bool(
            self.config.max_wall_seconds
            and self._train_start_time
            and time.perf_counter() - self._train_start_time >= self.config.max_wall_seconds
        )

    def _attack_batch(self, images: torch.Tensor, labels: torch.Tensor, attack_name: str) -> torch.Tensor:
        if attack_name == "clean":
            return images
        train_attack = dict(self.config.train_attack or {})
        configured_name = str(train_attack.get("name", "")).lower()
        if self.config.defense.lower() in {"fgsm_at", "pgd_at"} and configured_name:
            attack_name = configured_name.replace("-", "_")
        if attack_name == "fgsm":
            config = build_attack_config("fgsm", eps=float(train_attack.get("eps", self.config.eps)))
        elif attack_name == "fgsm_rs":
            config = build_attack_config(
                "fgsm_rs",
                eps=float(train_attack.get("eps", self.config.eps)),
                alpha=float(train_attack.get("alpha", 10 / 255)),
                random_start=bool(train_attack.get("random_start", True)),
            )
        elif attack_name == "pgd":
            config = build_attack_config(
                "pgd",
                eps=float(train_attack.get("eps", self.config.eps)),
                steps=int(train_attack.get("steps", self.config.pgd_steps)),
                alpha=float(train_attack.get("alpha", self.config.pgd_alpha)),
                random_start=bool(train_attack.get("random_start", True)),
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
            return {"fgsm_rs": 1.0}
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
        progress = tqdm(
            self._limited_loader(self.train_loader, self.config.max_train_batches),
            total=self._loader_total(self.train_loader, self.config.max_train_batches),
            desc=f"epoch {epoch_zero_based + 1}/{self.config.epochs} train/{self.config.defense}",
            leave=False,
            dynamic_ncols=True,
            unit="batch",
            disable=not self.config.show_progress,
        )

        for batch_idx, (images, labels) in enumerate(progress):
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
            progress.set_postfix(
                loss=f"{total_loss / max(1, total):.4f}",
                acc=f"{total_acc / max(1, total):.4f}",
                attack=attack_name,
            )
            if self._wall_clock_expired():
                break

        return {
            "train_loss": total_loss / max(1, total),
            "train_acc": total_acc / max(1, total),
            "attack_mix": json.dumps(mix, sort_keys=True),
            "attack_counts": json.dumps(attack_counts, sort_keys=True),
        }

    @torch.no_grad()
    def evaluate_clean(self, epoch_zero_based: int | None = None) -> dict[str, float]:
        self.model.eval()
        total = 0
        total_loss = 0.0
        total_acc = 0.0
        desc = "val/clean"
        if epoch_zero_based is not None:
            desc = f"epoch {epoch_zero_based + 1}/{self.config.epochs} {desc}"
        progress = tqdm(
            self._limited_loader(self.val_loader, self.config.max_eval_batches),
            total=self._loader_total(self.val_loader, self.config.max_eval_batches),
            desc=desc,
            leave=False,
            dynamic_ncols=True,
            unit="batch",
            disable=not self.config.show_progress,
        )
        for images, labels in progress:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            logits = self.model(images)
            loss = F.cross_entropy(logits, labels)
            batch_size = labels.size(0)
            total += batch_size
            total_loss += loss.item() * batch_size
            total_acc += accuracy_from_logits(logits, labels) * batch_size
            progress.set_postfix(loss=f"{total_loss / max(1, total):.4f}", acc=f"{total_acc / max(1, total):.4f}")
        return {"val_loss": total_loss / max(1, total), "val_acc": total_acc / max(1, total)}

    def _resolved_best_criterion(self) -> str:
        if self.config.best_criterion != "auto":
            return self.config.best_criterion
        return "clean_val" if self.config.defense.lower() == "standard" else "robust_val"

    def evaluate_robust(self, epoch_zero_based: int | None = None) -> dict[str, float]:
        attack = AttackFactory.create(
            build_attack_config(
                "pgd20",
                eps=self.config.eps,
                alpha=self.config.pgd_alpha,
                steps=self.config.robust_val_steps,
                random_start=True,
            )
        )
        self.model.eval()
        total = 0
        total_acc = 0.0
        desc = "val/pgd7"
        if epoch_zero_based is not None:
            desc = f"epoch {epoch_zero_based + 1}/{self.config.epochs} {desc}"
        progress = tqdm(
            self._limited_loader(self.val_loader, self.config.max_eval_batches),
            total=self._loader_total(self.val_loader, self.config.max_eval_batches),
            desc=desc,
            leave=False,
            dynamic_ncols=True,
            unit="batch",
            disable=not self.config.show_progress,
        )
        for images, labels in progress:
            if self.config.robust_val_subset_size and total >= self.config.robust_val_subset_size:
                break
            if self.config.robust_val_subset_size:
                remaining = self.config.robust_val_subset_size - total
                images = images[:remaining]
                labels = labels[:remaining]
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            adv_images = attack(self.model, images, labels)
            with torch.no_grad():
                logits = self.model(adv_images)
            batch_size = labels.size(0)
            total += batch_size
            total_acc += accuracy_from_logits(logits, labels) * batch_size
            progress.set_postfix(acc=f"{total_acc / max(1, total):.4f}", steps=self.config.robust_val_steps)
        return {"pgd7_val_acc": total_acc / max(1, total)}

    def train(self) -> dict[str, Any]:
        start = time.perf_counter()
        self._train_start_time = start
        history: list[dict[str, Any]] = []
        co_pgd7_history: list[float] = []
        co_detected = False
        co_epoch: int | None = None
        best_clean_acc = -1.0
        best_metric_value = -1.0
        best_epoch = 0
        best_clean_epoch = 0
        best_robust_acc: float | None = None
        best_criterion = self._resolved_best_criterion()
        best_metric = "pgd7_val_acc" if best_criterion == "robust_val" else "val_acc"
        best_path = self.run_dir / "best.pt"
        self._log(
            "[train] "
            f"run={self.run_name} model={self.config.model_name} defense={self.config.defense} "
            f"dataset={self.config.dataset_name} mode={self.config.mode} device={self.device} "
            f"epochs={self.config.epochs} batch_size={self.config.batch_size or 'loader'} "
            f"amp={bool(self.config.amp and self.device.type == 'cuda')} output={self.run_dir}"
        )
        if self.config.max_wall_seconds:
            self._log(f"[train] wall-clock budget={format_seconds(self.config.max_wall_seconds)}")
        epoch_progress = tqdm(
            range(self.config.epochs),
            desc=f"epochs/{self.run_name}",
            total=self.config.epochs,
            unit="epoch",
            dynamic_ncols=True,
            disable=not self.config.show_progress,
        )
        for epoch in epoch_progress:
            train_stats = self.train_one_epoch(epoch)
            val_stats = self.evaluate_clean(epoch)
            robust_stats: dict[str, float] = {}
            if best_criterion == "robust_val" and (epoch + 1) % max(1, self.config.eval_every) == 0:
                robust_stats = self.evaluate_robust(epoch)
            if self.config.co_check_enabled and (epoch + 1) % max(1, self.config.co_check_every) == 0:
                co_value = robust_stats.get("pgd7_val_acc")
                if co_value is None:
                    co_value = self.evaluate_robust(epoch)["pgd7_val_acc"]
                co_pgd7_history.append(co_value)
                if len(co_pgd7_history) >= 2 and co_pgd7_history[-2] - co_pgd7_history[-1] > self.config.co_threshold:
                    co_detected = True
                    co_epoch = epoch + 1
            self.scheduler.step()
            elapsed = time.perf_counter() - start
            row = {
                "epoch": epoch + 1,
                "defense": self.config.defense,
                "lr": self.optimizer.param_groups[0]["lr"],
                "wall_time_sec": elapsed,
                **train_stats,
                **val_stats,
                **robust_stats,
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
            if val_stats["val_acc"] > best_clean_acc:
                best_clean_acc = val_stats["val_acc"]
                best_clean_epoch = epoch + 1
                save_checkpoint(self.run_dir / "best_clean.pt", checkpoint)
            current_metric = robust_stats.get(best_metric) if best_criterion == "robust_val" else val_stats["val_acc"]
            best_updated = False
            if current_metric is not None and current_metric > best_metric_value:
                best_metric_value = current_metric
                best_epoch = epoch + 1
                if best_criterion == "robust_val":
                    best_robust_acc = current_metric
                save_checkpoint(best_path, checkpoint)
                best_updated = True
            epoch_progress.set_postfix(
                train_acc=self._fmt(train_stats.get("train_acc")),
                val_acc=self._fmt(val_stats.get("val_acc")),
                best=self._fmt(best_metric_value if best_metric_value >= 0 else None),
            )
            if self.config.progress_log_every and (epoch + 1) % max(1, self.config.progress_log_every) == 0:
                robust_part = ""
                if robust_stats:
                    robust_part = f" pgd7_val_acc={self._fmt(robust_stats.get('pgd7_val_acc'))}"
                self._log(
                    f"[epoch {epoch + 1:03d}/{self.config.epochs:03d}] "
                    f"train_loss={self._fmt(train_stats.get('train_loss'))} "
                    f"train_acc={self._fmt(train_stats.get('train_acc'))} "
                    f"val_loss={self._fmt(val_stats.get('val_loss'))} "
                    f"val_acc={self._fmt(val_stats.get('val_acc'))}"
                    f"{robust_part} best_{best_metric}={self._fmt(best_metric_value)} "
                    f"best_epoch={best_epoch or 'n/a'} best_updated={best_updated} "
                    f"elapsed={format_seconds(elapsed)}"
                )
            if co_detected:
                self._log(f"[train] catastrophic overfitting detected at epoch {co_epoch}; stopping early.")
                break
            if self._wall_clock_expired():
                self._log(f"[train] wall-clock budget reached after epoch {epoch + 1}; stopping.")
                break

        elapsed = time.perf_counter() - start
        device_count = torch.cuda.device_count() if self.device.type == "cuda" else 1
        gpu_name = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "cpu"
        metrics = {
            "exp_id": self.run_name,
            "model": self.config.model_name,
            "defense": self.config.defense,
            "dataset": self.config.dataset_name,
            "dataset_name": self.config.dataset_name,
            "mode": self.config.mode,
            "seed": self.config.seed,
            "clean_acc": best_clean_acc,
            "best_clean_acc": best_clean_acc,
            "best_clean_epoch": best_clean_epoch,
            "best_criterion": best_criterion,
            "best_metric": best_metric,
            "best_metric_value": best_metric_value,
            "best_epoch": best_epoch,
            "best_robust_acc": best_robust_acc,
            "train_time_sec": elapsed,
            "train_time_gpu_hours": gpu_hours_from_seconds(elapsed, device_count=device_count),
            "params": sum(parameter.numel() for parameter in self.model.parameters()),
            "checkpoint_path": str(best_path),
            "device": str(self.device),
            "max_train_batches": self.config.max_train_batches,
            "max_eval_batches": self.config.max_eval_batches,
            "gpu_name": gpu_name,
            "batch_size": self.config.batch_size,
            "amp": bool(self.config.amp and self.device.type == "cuda"),
            "session_id": self.session_id,
            "max_wall_seconds": self.config.max_wall_seconds,
            "wall_clock_stopped": self._wall_clock_expired(),
            "experiment_group": self.config.experiment_group,
            "train_attack_config": self.config.train_attack or {},
            "eval_attacks": self.config.eval_attacks or [],
            "co_detected": co_detected,
            "co_epoch": co_epoch,
            "co_pgd7_history": co_pgd7_history,
        }
        save_json(self.run_dir / "metrics.json", metrics)
        self._log(
            "[train] finished "
            f"run={self.run_name} best_{best_metric}={self._fmt(best_metric_value)} "
            f"best_epoch={best_epoch or 'n/a'} elapsed={format_seconds(elapsed)} "
            f"best={best_path} metrics={self.run_dir / 'metrics.json'}"
        )
        return metrics
