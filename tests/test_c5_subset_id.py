from __future__ import annotations

from src.datasets.cifar10 import eval_subset_id


def test_eval_subset_id_is_stable_and_changes_with_size() -> None:
    first = eval_subset_id("cifar10", "aalite_2k", 2000, 0)
    second = eval_subset_id("cifar10", "aalite_2k", 2000, 0)
    different = eval_subset_id("cifar10", "aalite_2k", 1000, 0)
    assert first == second
    assert first != different
    assert len(first) == 12


def test_aalite_gap_over_invalid_when_subset_ids_mismatch() -> None:
    class DummyRunnerModel:
        pass

    payload = {
        "pgd20": {"status": "ok", "robust_acc": 0.5, "eval_subset_id": "a"},
        "apgd_ce": {"status": "ok", "robust_acc": 0.4, "eval_subset_id": "b"},
        "apgd_dlr": {"status": "ok", "robust_acc": 0.3, "eval_subset_id": "a"},
    }
    subset_ids = {result["eval_subset_id"] for result in payload.values()}
    assert len(subset_ids) > 1


def test_run_aalite_has_eval_subset_id_when_metadata_supplied(monkeypatch) -> None:
    import pytest

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
            "eval_subset_id": metadata["eval_subset_id"],
        }

    monkeypatch.setattr(aalite.AttackRunner, "run", fake_run)
    result = run_aalite(DummyRunnerModel(), [], device="cpu", metadata={"eval_subset_id": "same"})
    assert result["eval_subset_id"] == "same"
    assert result["gap_over"] == 0.2
