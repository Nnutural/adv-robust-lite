# Ready for Real Run

## Smoke to Real Switch

- Use CIFAR-10 real mode: `--dataset cifar10 --mode real --download`.
- Aggregate only real outputs: `python scripts/aggregate_results.py --input results/real/raw --output results/real/tables`.
- Do not use `--allow-smoke` for paper tables.
- Use the real configs: `configs/experiments/g1_structure_real.yaml`, `g2_defense_real.yaml`, `g3_eval_protocol_real.yaml`, `g3_traps_real.yaml`, `g4_mixed_ablation_real.yaml`, `g5_budget_real.yaml`.
- Recommended real training defaults: `epochs=80`, `batch_size=128` if memory allows, `eps=8/255`, `pgd_at train_attack.steps=7`, `pgd20 eval steps=20`, `apgd_ce/apgd_dlr steps=50`.
- For AA-Lite use the fixed subset config `configs/eval/subsets.yaml` with `aalite_2k`; PGD-20 full test can use `--subset-size 0`.

## Compute Reference

- Reference scale only: PreActResNet-18 + PGD-7-AT + 80 epochs on a common single GPU is expected to take hours, not minutes.
- On mid-range consumer GPUs, expect roughly a several-hour to half-day wall-clock range depending on batch size, dataloader speed, AMP, and APGD validation frequency.
- All latency, wall-clock, and GPU-hour values are待 real 运行后回填.

## Recommended Order

1. G1 real standard baselines: SmallCNN, ResNet-18, MobileNetV2 with FGSM, PGD-20, APGD-CE-50, APGD-DLR-50.
2. G2 real defense matrix on PreActResNet-18: Standard, FGSM-AT, PGD-AT, Fixed Mixed-AT.
3. G3 protocol checks: honest models, Trap-A, Trap-B, AA-Lite, Square subset, and optional AutoAttack subset calibration.
4. G4 Fixed Mixed-AT ablation after G2 checkpoints are stable.
5. G5 equal epoch and equal GPU-hour comparison in one continuous session per comparison group, keeping GPU, batch size, AMP, and session fields aligned.

## Out of Scope Here

- Full Budget-aware Scheduler implementation.
- MI-FGSM transfer attack.
- ViT experiments.
- VGG-16 experiments.
- Full AutoAttack as part of the main matrix.
