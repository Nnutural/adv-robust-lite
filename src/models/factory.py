from __future__ import annotations

from typing import Any

from .mobilenet import mobilenet_v2
from .normalize_wrapper import NormalizeWrapper
from .preact_resnet import preact_resnet18
from .resnet import resnet18
from .smallcnn import small_cnn

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def build_model(
    model_name: str,
    num_classes: int = 10,
    normalize: bool = True,
    mean: tuple[float, float, float] | list[float] = CIFAR10_MEAN,
    std: tuple[float, float, float] | list[float] = CIFAR10_STD,
    **kwargs: Any,
):
    normalized_name = model_name.lower().replace("-", "_")
    if normalized_name == "smallcnn":
        backbone = small_cnn(num_classes=num_classes)
    elif normalized_name == "resnet18":
        backbone = resnet18(num_classes=num_classes)
    elif normalized_name == "mobilenetv2":
        backbone = mobilenet_v2(num_classes=num_classes, width_mult=float(kwargs.get("width_mult", 1.0)))
    elif normalized_name == "preact_resnet18":
        backbone = preact_resnet18(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if normalize:
        return NormalizeWrapper(backbone, mean=mean, std=std)
    return backbone


def build_model_from_config(cfg: dict[str, Any]):
    model_cfg = cfg.get("model", {})
    return build_model(
        model_name=model_cfg.get("name", "smallcnn"),
        num_classes=int(model_cfg.get("num_classes", 10)),
        normalize=bool(model_cfg.get("normalize", True)),
        mean=model_cfg.get("mean", CIFAR10_MEAN),
        std=model_cfg.get("std", CIFAR10_STD),
        width_mult=model_cfg.get("width_mult", 1.0),
    )
