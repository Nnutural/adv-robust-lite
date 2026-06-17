# Experiment Plan

## Threat Model

- Dataset: CIFAR-10.
- Norm: L-infinity.
- Main epsilon: `8/255`.
- Input range: `[0,1]`.
- Attack target: untargeted classification error.
- Normalization is inside `NormalizeWrapper`, after adversarial perturbation and clipping.

## Main Experiments

### G1 Structure Robustness

Models: `smallcnn`, `resnet18`, `mobilenetv2`.

Training: `standard`.

Attacks: `fgsm`, `pgd20`, `apgd_ce`, `apgd_dlr`.

Outputs:

- `results/tables/main_robustness.csv`
- `results/figures/fig2_structure_heatmap.{png,pdf}`

### G2 Defense Generalization

Model: `preact_resnet18`.

Defenses: `standard`, `fgsm_at`, `pgd_at`, `fixed_mixed_at`.

Attacks: `fgsm`, `pgd20`, `apgd_ce`, `apgd_dlr`.

Outputs:

- `results/tables/main_robustness.csv`
- `results/figures/fig3_defense_heatmap.{png,pdf}` if split by defense later

### G3 Evaluation Protocol Ablation

Protocols:

- PGD-only: `pgd20`
- PGD + APGD-CE
- AA-Lite: `pgd20 + apgd_ce + apgd_dlr`
- Full AutoAttack subset: fixed 1000-image test subset

Outputs:

- `results/tables/aa_subset_check.csv`
- `results/figures/fig4_gap_over.{png,pdf}`

### G4 Fixed Mixed-AT Ablation

Defenses:

- `fgsm_at`
- `pgd_at`
- `fixed_mixed_at`
- optional `budget_scheduler_at`

Outputs:

- `results/tables/budget_comparison.csv`
- `results/figures/fig5_cost_robust_tradeoff.{png,pdf}`

### G5 Equal Epoch vs Equal GPU Hours

Report both same-epoch results and GPU-hour-normalized comparison:

```text
RobustGain = R_lite(defense) - R_lite(standard)
Efficiency = RobustGain / GPU_hours
```

## Script Map

Training:

```bash
python scripts/train.py --model preact_resnet18 --defense fixed_mixed_at --seed 0
```

Single attack:

```bash
python scripts/evaluate_attack.py --checkpoint checkpoints/.../best.pt --model preact_resnet18 --attack pgd20
```

AA-Lite:

```bash
python scripts/run_aalite.py --checkpoint checkpoints/.../best.pt --model preact_resnet18
```

AutoAttack subset:

```bash
python scripts/run_autoattack_subset.py --checkpoint checkpoints/.../best.pt --model preact_resnet18 --subset-size 1000
```

Diagnostics:

```bash
python scripts/run_diagnostics.py --checkpoint checkpoints/.../best.pt --model preact_resnet18 --max-samples 1000
```

Aggregation and figures:

```bash
python scripts/aggregate_results.py --input results/raw --output results/tables
python scripts/make_figures.py --tables results/tables --output results/figures
```

## Paper Table/Figure Mapping

- Table 1: model clean accuracy, parameters, training time from `main_robustness.csv`.
- Table 2: attack robustness matrix from `main_robustness.csv`.
- Table 3: equal epoch and equal GPU-hour comparison from `budget_comparison.csv`.
- Table 4: gradient masking diagnostic flags from `gradient_masking_diagnostics.csv`.
- Figure 2: structure robustness heatmap.
- Figure 4: PGD-only vs AA-Lite and `Gap_over`.
- Figure 5: clean accuracy, `R_lite`, and GPU-hour trade-off.
- Figure 6: epsilon monotonicity curve.

## Execution Policy

Run smoke first with `--epochs 1 --max-samples 256`. Do not launch the full main matrix until the smoke chain writes JSON, CSV, and at least one figure successfully.

