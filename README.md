# adv-robust-lite

This project is a reproducible CIFAR-10 adversarial robustness framework for the course project:

**Lightweight reliable evaluation and budget-aware adversarial training for image classification robustness.**

It focuses on a practical smoke-to-main-experiment path, not SOTA performance.

## Smoke/Real Separation

P6 and development smoke outputs are engineering checks only and must not be used as paper results. Fake-data runs require `--mode smoke` and write under `results/smoke/`. Paper-facing runs use `--mode real` with CIFAR-10 and write under `results/real/`; real aggregation rejects `mode=smoke` or `dataset_name=fake_cifar10` unless `--allow-smoke` is explicitly set for development.

## Research Questions

1. Do SmallCNN, ResNet-18, and MobileNetV2 show consistent architectural vulnerability under FGSM, PGD, and APGD attacks?
2. Do FGSM-AT, PGD-AT, and Fixed Mixed-AT generalize beyond the attack used during training?
3. Does AA-Lite reduce PGD-only robustness overestimation?
4. Under equal epoch and equal GPU-hour views, which defense gives the best clean/robustness/cost trade-off?

## Install

```bash
pip install -r requirements.txt
```

`torchattacks` and `autoattack` are optional at runtime. If they are unavailable, FGSM/PGD use local implementations and APGD uses a local APGD-like fallback. Full AutoAttack subset evaluation requires the `autoattack` package.

## Data

CIFAR-10 is loaded through `torchvision.datasets.CIFAR10`. Training transforms do not normalize images. Attacks operate on `[0,1]` pixel tensors; normalization is inside `NormalizeWrapper`.

## Minimal Commands

```bash
python scripts/train.py --model smallcnn --defense standard --epochs 1 --dataset fake_cifar10 --mode smoke --train-subset-size 256 --val-subset-size 128 --max-train-batches 2 --max-eval-batches 1
python scripts/evaluate_clean.py --checkpoint checkpoints/smallcnn_standard_seed0/best.pt --model smallcnn --dataset fake_cifar10 --mode smoke --subset-size 128 --max-eval-batches 1
python scripts/evaluate_attack.py --checkpoint checkpoints/smallcnn_standard_seed0/best.pt --model smallcnn --attack fgsm --dataset fake_cifar10 --mode smoke --subset-size 128 --max-eval-batches 1
python scripts/run_aalite.py --checkpoint checkpoints/smallcnn_standard_seed0/best.pt --model smallcnn --dataset fake_cifar10 --mode smoke --subset-size 128 --max-eval-batches 1
python scripts/aggregate_results.py --input results/smoke/raw --output results/smoke/tables --allow-smoke
python scripts/make_figures.py --tables results/smoke/tables --output results/smoke/figures
```

## Full Main Matrix

Structure comparison:

```bash
python scripts/train.py --model smallcnn --defense standard --seed 0 --download
python scripts/train.py --model resnet18 --defense standard --seed 0 --download
python scripts/train.py --model mobilenetv2 --defense standard --seed 0 --download
```

Defense comparison:

```bash
python scripts/train.py --model preact_resnet18 --defense standard --seed 0 --download
python scripts/train.py --model preact_resnet18 --defense fgsm_at --seed 0 --download
python scripts/train.py --model preact_resnet18 --defense pgd_at --seed 0 --download
python scripts/train.py --model preact_resnet18 --defense fixed_mixed_at --seed 0 --download
```

Evaluation:

```bash
python scripts/run_aalite.py --checkpoint checkpoints/preact_fixed_mixed_seed0/best.pt --model preact_resnet18
python scripts/run_autoattack_subset.py --checkpoint checkpoints/preact_fixed_mixed_seed0/best.pt --model preact_resnet18 --subset-size 1000
python scripts/run_diagnostics.py --checkpoint checkpoints/preact_fixed_mixed_seed0/best.pt --model preact_resnet18 --max-samples 1000
```

## AA-Lite Definition

AA-Lite is a project-defined lightweight protocol, not official AutoAttack and not a replacement for full AutoAttack.

`R_lite` is white-box only. It is the worst robust accuracy over PGD-20, APGD-CE, and APGD-DLR; it does not include black-box attacks. Square is evaluated separately on a fixed subset and used with black-box sanity diagnostics.

```text
R_lite = min(Acc_PGD20, Acc_APGD_CE, Acc_APGD_DLR)
Gap_over = Acc_PGD20 - R_lite
Bias_AA = R_lite_subset - Acc_AA_subset
Dev_AA = abs(R_lite_subset - Acc_AA_subset)
```

## Output Fields

`results/tables/main_robustness.csv`:

`exp_id, model, defense, seed, clean_acc, fgsm_acc, pgd20_acc, apgd_ce_acc, apgd_dlr_acc, r_lite, gap_over, train_time_sec, train_time_gpu_hours, checkpoint_path`

`results/tables/aa_subset_check.csv`:

`exp_id, model, defense, r_lite_subset, aa_subset_acc, bias_aa, dev_aa, square_acc, subset_size`

`results/tables/gradient_masking_diagnostics.csv`:

`exp_id, eps_monotone, step_monotone, restart_stable, blackbox_not_stronger, diagnosis_pass, notes`

## Figures

```bash
python scripts/make_figures.py --tables results/tables --output results/figures
```

Generated figures include the robustness heatmap, PGD-only vs AA-Lite comparison, cost/robustness trade-off, and epsilon curve when diagnostic raw JSON is available.

## Known Limits

The local APGD fallback is for robust smoke execution when `torchattacks` is absent. Formal results should install `torchattacks` and use full AutoAttack only on the fixed subset as a calibration reference.
