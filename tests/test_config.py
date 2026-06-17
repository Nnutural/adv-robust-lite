from pathlib import Path
import subprocess
import sys

from utils.config import deep_merge, load_config, parse_scalar, set_by_dotted_key


def test_deep_merge_preserves_nested_values() -> None:
    merged = deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}})
    assert merged == {"a": {"b": 3, "c": 2}}


def test_base_config_loads() -> None:
    cfg = load_config(Path("configs/base.yaml"))
    assert cfg["project"]["name"] == "adv_robust_lite"
    assert cfg["threat_model"]["input_space"] == "pixel_0_1"


def test_dotted_override_and_parse_scalar() -> None:
    cfg = set_by_dotted_key({}, "training.epochs", parse_scalar("1"))
    assert cfg["training"]["epochs"] == 1
    assert parse_scalar("false") is False


def test_low_budget_cli_flags_are_available() -> None:
    commands = {
        "scripts/train.py": [
            "--train-subset-size",
            "--val-subset-size",
            "--max-train-batches",
            "--max-eval-batches",
            "--batch-size",
            "--num-workers",
            "--device",
            "--epochs",
        ],
        "scripts/evaluate_clean.py": ["--subset-size", "--max-eval-batches", "--batch-size", "--num-workers", "--device"],
        "scripts/evaluate_attack.py": ["--subset-size", "--max-eval-batches", "--batch-size", "--num-workers", "--device"],
        "scripts/run_aalite.py": ["--subset-size", "--max-eval-batches", "--batch-size", "--num-workers", "--device"],
        "scripts/run_autoattack_subset.py": ["--subset-size", "--max-eval-batches", "--batch-size", "--num-workers", "--device"],
    }
    for script, expected_flags in commands.items():
        result = subprocess.run([sys.executable, script, "--help"], capture_output=True, text=True, check=True)
        help_text = result.stdout
        for flag in expected_flags:
            assert flag in help_text
