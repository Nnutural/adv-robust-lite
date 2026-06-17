from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SchedulerSignal:
    clean_acc_val: float
    fgsm_acc_val: float
    pgd_acc_val: float
    apgd_dlr_acc_val: float
    eps_monotone: bool = True
    gpu_hours_left: float = 999.0


class BudgetAwareScheduler:
    """Lightweight reserved interface for budget-aware attack mix updates."""

    def __init__(
        self,
        mix: dict[str, float] | None = None,
        tau_fgsm_pgd: float = 0.05,
        tau_pgd_apgd: float = 0.03,
        tau_clean_drop: float = 0.08,
        min_gpu_hours_left: float = 0.25,
    ) -> None:
        self.mix = mix or {"clean": 0.5, "fgsm": 0.5, "pgd": 0.0, "apgd_ce": 0.0}
        self.initial_clean_acc: float | None = None
        self.tau_fgsm_pgd = tau_fgsm_pgd
        self.tau_pgd_apgd = tau_pgd_apgd
        self.tau_clean_drop = tau_clean_drop
        self.min_gpu_hours_left = min_gpu_hours_left

    def update(self, signal: SchedulerSignal) -> dict[str, float]:
        if self.initial_clean_acc is None:
            self.initial_clean_acc = signal.clean_acc_val
        if not signal.eps_monotone:
            return self._normalized(self.mix)
        if signal.gpu_hours_left < self.min_gpu_hours_left:
            self.mix["apgd_ce"] = 0.0
            self.mix["pgd"] = min(0.5, self.mix.get("pgd", 0.0))
            self.mix["fgsm"] = max(self.mix.get("fgsm", 0.0), 0.3)
        if signal.fgsm_acc_val - signal.pgd_acc_val > self.tau_fgsm_pgd:
            self.mix["pgd"] = self.mix.get("pgd", 0.0) + 0.1
            self.mix["fgsm"] = max(0.0, self.mix.get("fgsm", 0.0) - 0.1)
        if signal.pgd_acc_val - signal.apgd_dlr_acc_val > self.tau_pgd_apgd:
            self.mix["apgd_ce"] = self.mix.get("apgd_ce", 0.0) + 0.05
            self.mix["pgd"] = max(0.0, self.mix.get("pgd", 0.0) - 0.05)
        if self.initial_clean_acc - signal.clean_acc_val > self.tau_clean_drop:
            self.mix["clean"] = self.mix.get("clean", 0.0) + 0.1
            strongest = "apgd_ce" if self.mix.get("apgd_ce", 0.0) > 0 else "pgd"
            self.mix[strongest] = max(0.0, self.mix.get(strongest, 0.0) - 0.1)
        return self._normalized(self.mix)

    @staticmethod
    def _normalized(mix: dict[str, float]) -> dict[str, float]:
        total = sum(max(0.0, value) for value in mix.values())
        if total <= 0:
            return {"clean": 1.0}
        return {key: max(0.0, value) / total for key, value in mix.items()}

