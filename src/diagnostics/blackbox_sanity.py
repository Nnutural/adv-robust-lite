from __future__ import annotations


def blackbox_not_stronger(whitebox_acc: float, blackbox_acc: float, tolerance: float = 0.02) -> bool:
    """Return true when black-box attack is not systematically stronger than white-box."""
    return blackbox_acc >= whitebox_acc - tolerance

