from __future__ import annotations

from src.attacks.factory import AttackConfig, AttackUnavailableError


def square_attack(model, images, labels, config: AttackConfig):
    try:
        import torchattacks
    except ModuleNotFoundError as exc:
        raise AttackUnavailableError(
            "torchattacks is not installed; Square is skipped instead of using a non-official fallback."
        ) from exc

    if not hasattr(torchattacks, "Square"):
        raise AttackUnavailableError("torchattacks.Square is unavailable in this environment.")
    attack = torchattacks.Square(model, norm="Linf", eps=config.eps, n_queries=getattr(config, "n_queries", 2000))
    return attack(images, labels).clamp(0.0, 1.0).detach()
