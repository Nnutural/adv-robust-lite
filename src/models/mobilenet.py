from __future__ import annotations


def mobilenet_v2(num_classes: int = 10, width_mult: float = 1.0):
    from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

    _ = MobileNet_V2_Weights
    model = mobilenet_v2(weights=None, num_classes=num_classes, width_mult=width_mult)
    model.features[0][0].stride = (1, 1)
    return model
