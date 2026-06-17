from __future__ import annotations

import pytest

from src.datasets.cifar10 import eval_subset_id


class _DummyModel:
    pass


def test_eval_subset_id_is_stable_and_changes_with_size() -> None:
    first = eval_subset_id("cifar10", "aalite_2k", 2000, 0)
    second = eval_subset_id("cifar10", "aalite_2k", 2000, 0)
    different = eval_subset_id("cifar10", "aalite_2k", 1000, 0)
    assert first == second
    assert first != different
    assert len(first) == 12


def test_aalite_gap_over_invalid_when_subset_ids_mismatch(monkeypatch) -> None:
    pytest.importorskip("torch")
    import src.attacks.aalite as aalite
    from src.attacks.aalite import run_aalite

    per_attack_subset = {"pgd20": "a", "apgd_ce": "b", "apgd_dlr": "a"}

    def fake_run(self, model, dataloader, metadata=None):
        attack = self.attack_config.name
        return {
            "attack": attack,
            "status": "ok",
            "robust_acc": {"pgd20": 0.5, "apgd_ce": 0.4, "apgd_dlr": 0.3}[attack],
            "clean_acc": 0.6,
            "num_samples": 1,
            "eval_subset_id": per_attack_subset[attack],
        }

    monkeypatch.setattr(aalite.AttackRunner, "run", fake_run)
    result = run_aalite(_DummyModel(), [], device="cpu", metadata={"eval_subset_id": "a"})
    assert result["r_lite"] is None
    assert result["gap_over"] is None
    assert result["gap_over_error"] == "subset_id_mismatch"


def test_aalite_gap_over_ok_when_subset_ids_match(monkeypatch) -> None:
    pytest.importorskip("torch")
    import src.attacks.aalite as aalite
    from src.attacks.aalite import run_aalite

    def fake_run(self, model, dataloader, metadata=None):
        attack = self.attack_config.name
        return {
            "attack": attack,
            "status": "ok",
            "robust_acc": {"pgd20": 0.5, "apgd_ce": 0.4, "apgd_dlr": 0.3}[attack],
            "clean_acc": 0.6,
            "num_samples": 1,
            "eval_subset_id": "same",
        }

    monkeypatch.setattr(aalite.AttackRunner, "run", fake_run)
    result = run_aalite(_DummyModel(), [], device="cpu", metadata={"eval_subset_id": "same"})
    assert result["eval_subset_id"] == "same"
    assert result["gap_over"] == pytest.approx(0.2, abs=1e-9)
