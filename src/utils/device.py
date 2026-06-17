from __future__ import annotations

from typing import Any


def get_device(preference: str = "auto"):
    import torch

    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(preference)


def get_device_info() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info.update(
            {
                "cuda_device": idx,
                "gpu_name": props.name,
                "total_memory_bytes": props.total_memory,
            }
        )
    return info
