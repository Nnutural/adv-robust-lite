from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn


class AttackUnavailableError(RuntimeError):
    """Raised when an optional attack dependency is not available."""


@dataclass
class AttackConfig:
    name: str
    eps: float = 8 / 255
    alpha: float = 2 / 255
    steps: int = 20
    restarts: int = 1
    random_start: bool = True
    loss: str = "ce"
    eot_samples: int = 0
    n_queries: int = 2000


@contextmanager
def temporary_eval(model: nn.Module, enabled: bool = True):
    previous_training_state = model.training
    if enabled:
        model.eval()
    try:
        yield
    finally:
        if enabled:
            model.train(previous_training_state)


def dlr_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    sorted_logits, sorted_indices = logits.sort(dim=1)
    y_logit = logits.gather(1, labels.view(-1, 1)).squeeze(1)
    top1 = sorted_logits[:, -1]
    top2 = sorted_logits[:, -2]
    top3 = sorted_logits[:, -3]
    top1_is_label = sorted_indices[:, -1] == labels
    best_other = torch.where(top1_is_label, top2, top1)
    denominator = (top1 - top3).clamp_min(1e-12)
    return -((y_logit - best_other) / denominator).mean()


def _loss(logits: torch.Tensor, labels: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name == "dlr":
        return dlr_loss(logits, labels)
    return F.cross_entropy(logits, labels)


def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 8 / 255,
    set_eval: bool = True,
) -> torch.Tensor:
    if eps <= 0:
        return images.detach()
    images = images.detach()
    with temporary_eval(model, enabled=set_eval):
        adv = images.clone().detach().requires_grad_(True)
        logits = model(adv)
        loss = F.cross_entropy(logits, labels)
        grad = torch.autograd.grad(loss, adv, only_inputs=True)[0]
        adv = adv + eps * grad.sign()
    return adv.clamp(0.0, 1.0).detach()


def fgsm_rs_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 8 / 255,
    alpha: float = 10 / 255,
    random_start: bool = True,
    set_eval: bool = True,
) -> torch.Tensor:
    images = images.detach()
    with temporary_eval(model, enabled=set_eval):
        if random_start:
            adv = (images + torch.empty_like(images).uniform_(-eps, eps)).clamp(0.0, 1.0)
        else:
            adv = images.clone()
        adv = adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(adv), labels)
        grad = torch.autograd.grad(loss, adv, only_inputs=True)[0]
        adv = adv + alpha * grad.sign()
        delta = (adv - images).clamp(-eps, eps)
        adv = (images + delta).clamp(0.0, 1.0)
    return adv.detach()


def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 8 / 255,
    alpha: float = 2 / 255,
    steps: int = 20,
    restarts: int = 1,
    random_start: bool = True,
    loss_name: str = "ce",
    set_eval: bool = True,
    eot_samples: int = 0,
) -> torch.Tensor:
    if eps <= 0 or steps <= 0:
        return images.detach()

    images = images.detach()
    best_adv = images.clone()
    best_loss = torch.full((images.size(0),), -float("inf"), device=images.device)

    with temporary_eval(model, enabled=set_eval):
        for _ in range(max(1, restarts)):
            if random_start:
                adv = images + torch.empty_like(images).uniform_(-eps, eps)
                adv = adv.clamp(0.0, 1.0)
            else:
                adv = images.clone()

            for _step in range(steps):
                adv = adv.detach().requires_grad_(True)
                if eot_samples and eot_samples > 0:
                    from src.attacks.eot import eot_grad

                    grad = eot_grad(model, adv, labels, lambda logits, y: _loss(logits, y, loss_name), eot_samples)
                else:
                    logits = model(adv)
                    loss = _loss(logits, labels, loss_name)
                    grad = torch.autograd.grad(loss, adv, only_inputs=True)[0]
                adv = adv + alpha * grad.sign()
                delta = (adv - images).clamp(-eps, eps)
                adv = (images + delta).clamp(0.0, 1.0)

            with torch.no_grad():
                logits = model(adv)
                per_sample = F.cross_entropy(logits, labels, reduction="none")
                replace = per_sample > best_loss
                best_loss[replace] = per_sample[replace]
                best_adv[replace] = adv[replace]

    return best_adv.detach()


def _torchattacks_apgd(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    config: AttackConfig,
) -> torch.Tensor:
    try:
        import torchattacks
    except ModuleNotFoundError as exc:
        raise AttackUnavailableError(
            "torchattacks is not installed; APGD is skipped instead of using a non-official fallback."
        ) from exc

    try:
        attack = torchattacks.APGD(
            model,
            norm="Linf",
            eps=config.eps,
            steps=config.steps,
            n_restarts=config.restarts,
            loss=config.loss,
        )
        attack.set_model_training_mode(model_training=False, batchnorm_training=False, dropout_training=False)
        adv = attack(images, labels)
        return adv.clamp(0.0, 1.0).detach()
    except Exception as exc:
        raise RuntimeError(f"torchattacks APGD failed: {exc}") from exc


class AttackFactory:
    @staticmethod
    def create(config: AttackConfig | dict) -> Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]:
        if isinstance(config, dict):
            config = AttackConfig(**config)
        name = config.name.lower().replace("-", "_")
        if name == "fgsm":
            return lambda model, x, y: fgsm_attack(model, x, y, eps=config.eps)
        if name == "fgsm_rs":
            return lambda model, x, y: fgsm_rs_attack(
                model,
                x,
                y,
                eps=config.eps,
                alpha=config.alpha,
                random_start=config.random_start,
            )
        if name in {"pgd", "pgd20"}:
            return lambda model, x, y: pgd_attack(
                model,
                x,
                y,
                eps=config.eps,
                alpha=config.alpha,
                steps=config.steps,
                restarts=config.restarts,
                random_start=config.random_start,
                loss_name="ce",
                eot_samples=config.eot_samples,
            )
        if name in {"apgd_ce", "apgd_dlr"}:
            loss = "dlr" if name.endswith("dlr") else "ce"

            def _run(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                apgd_config = AttackConfig(**{**config.__dict__, "loss": loss})
                return _torchattacks_apgd(model, x, y, apgd_config)

            return _run
        if name == "square":
            def _run_square(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                from src.attacks.square import square_attack

                return square_attack(model, x, y, config)

            return _run_square
        raise ValueError(f"Unsupported attack: {config.name}")


def build_attack_config(name: str, eps: float = 8 / 255, steps: int | None = None, **kwargs) -> AttackConfig:
    normalized = name.lower().replace("-", "_")
    defaults = {
        "fgsm": {"steps": 1, "alpha": eps},
        "fgsm_rs": {"steps": 1, "alpha": 10 / 255, "random_start": True},
        "pgd": {"steps": 20, "alpha": 2 / 255},
        "pgd20": {"steps": 20, "alpha": 2 / 255},
        "apgd_ce": {"steps": 50, "alpha": 2 / 255, "loss": "ce"},
        "apgd_dlr": {"steps": 50, "alpha": 2 / 255, "loss": "dlr"},
        "square": {"steps": 1, "alpha": 2 / 255, "loss": "ce", "n_queries": 2000},
    }
    if normalized not in defaults:
        raise ValueError(f"Unsupported attack: {name}")
    payload = {"name": normalized, "eps": eps, **defaults[normalized], **kwargs}
    if steps is not None:
        payload["steps"] = steps
    return AttackConfig(**payload)
