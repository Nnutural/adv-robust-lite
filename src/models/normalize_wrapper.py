from __future__ import annotations


class NormalizeWrapper:
    """Wrap a backbone so callers pass images in [0,1] pixel space."""

    def __new__(cls, backbone, mean, std):
        import torch
        from torch import nn

        class _NormalizeWrapper(nn.Module):
            def __init__(self, wrapped_backbone, wrapped_mean, wrapped_std):
                super().__init__()
                self.backbone = wrapped_backbone
                mean_tensor = torch.tensor(wrapped_mean, dtype=torch.float32).view(1, -1, 1, 1)
                std_tensor = torch.tensor(wrapped_std, dtype=torch.float32).view(1, -1, 1, 1)
                self.register_buffer("mean", mean_tensor)
                self.register_buffer("std", std_tensor)

            def forward(self, x):
                x = (x - self.mean) / self.std
                return self.backbone(x)

        return _NormalizeWrapper(backbone, mean, std)
