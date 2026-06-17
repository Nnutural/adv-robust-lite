from __future__ import annotations


def seconds_to_gpu_hours(seconds: float, gpu_count: int = 1) -> float:
    return seconds * max(gpu_count, 1) / 3600.0


def robust_gain(r_lite_defense: float, r_lite_standard: float) -> float:
    return r_lite_defense - r_lite_standard


def gain_per_gpu_hour(*args: float) -> float:
    """Support both (gain, gpu_hours) and (r_lite_defense, r_lite_standard, gpu_hours)."""
    if len(args) == 2:
        gain, gpu_hours = args
    elif len(args) == 3:
        r_lite_defense, r_lite_standard, gpu_hours = args
        gain = robust_gain(r_lite_defense, r_lite_standard)
    else:
        raise TypeError("gain_per_gpu_hour expects 2 or 3 numeric arguments")
    return 0.0 if gpu_hours <= 0 else gain / gpu_hours


def clean_drop(clean_acc_standard: float, clean_acc_defense: float) -> float:
    return clean_acc_standard - clean_acc_defense
