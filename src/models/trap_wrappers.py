from __future__ import annotations


class LogitScaleWrapper:
    def __new__(cls, backbone, scale: float = 50.0):
        from torch import nn

        class _LogitScaleWrapper(nn.Module):
            def __init__(self, wrapped_backbone, wrapped_scale: float):
                super().__init__()
                self.backbone = wrapped_backbone
                self.scale = float(wrapped_scale)

            def forward(self, x):
                return self.backbone(x) * self.scale

        return _LogitScaleWrapper(backbone, scale)


class InputRandomizationWrapper:
    def __new__(cls, backbone, kind: str = "pad_resize", pad: int = 4, out_size: int = 32, jitter: float = 0.0):
        import torch
        import torch.nn.functional as F
        from torch import nn

        class _InputRandomizationWrapper(nn.Module):
            def __init__(self, wrapped_backbone, wrapped_kind: str, wrapped_pad: int, wrapped_out_size: int, wrapped_jitter: float):
                super().__init__()
                self.backbone = wrapped_backbone
                self.kind = wrapped_kind
                self.pad = int(wrapped_pad)
                self.out_size = int(wrapped_out_size)
                self.jitter = float(wrapped_jitter)

            def _pad_resize(self, x):
                if self.pad <= 0:
                    return x
                padded = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="reflect")
                max_offset = self.pad * 2
                top = int(torch.randint(0, max_offset + 1, (1,), device=x.device).item())
                left = int(torch.randint(0, max_offset + 1, (1,), device=x.device).item())
                return padded[:, :, top : top + self.out_size, left : left + self.out_size].clamp(0.0, 1.0)

            def _quantize(self, x):
                levels = 16
                if self.jitter > 0:
                    x = (x + torch.empty_like(x).uniform_(-self.jitter, self.jitter)).clamp(0.0, 1.0)
                return torch.round(x * (levels - 1)) / (levels - 1)

            def forward(self, x):
                if self.kind == "quantize":
                    x = self._quantize(x)
                else:
                    x = self._pad_resize(x)
                return self.backbone(x)

        return _InputRandomizationWrapper(backbone, kind, pad, out_size, jitter)
