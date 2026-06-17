# 实验计划

## 威胁模型

- 数据集：CIFAR-10。
- 范数约束：L-infinity。
- 主要扰动预算：`8/255`。
- 输入范围：`[0,1]`。
- 攻击目标：非定向分类错误。
- 标准化位置：对抗扰动和裁剪之后，在 `NormalizeWrapper` 内部完成标准化。

## 主实验

### G1 结构鲁棒性实验

模型：

- `smallcnn`
- `resnet18`
- `mobilenetv2`

训练方式：

- `standard`

攻击：

- `fgsm`
- `pgd20`
- `apgd_ce`
- `apgd_dlr`

输出：

- `results/tables/main_robustness.csv`
- `results/figures/fig2_structure_heatmap.{png,pdf}`

### G2 防御泛化实验

模型：

- `preact_resnet18`

防御方法：

- `standard`
- `fgsm_at`
- `pgd_at`
- `fixed_mixed_at`

攻击：

- `fgsm`
- `pgd20`
- `apgd_ce`
- `apgd_dlr`

输出：

- `results/tables/main_robustness.csv`
- 如果后续按 defense 拆分出图，则输出 `results/figures/fig3_defense_heatmap.{png,pdf}`

### G3 评测协议消融

协议：

- PGD-only：`pgd20`
- PGD + APGD-CE
- AA-Lite：`pgd20 + apgd_ce + apgd_dlr`
- 完整 AutoAttack 子集：固定 1000 张测试图像子集

输出：

- `results/tables/aa_subset_check.csv`
- `results/figures/fig4_gap_over.{png,pdf}`

### G4 Fixed Mixed-AT 组件消融

防御方法：

- `fgsm_at`
- `pgd_at`
- `fixed_mixed_at`
- 可选：`budget_scheduler_at`

输出：

- `results/tables/budget_comparison.csv`
- `results/figures/fig5_cost_robust_tradeoff.{png,pdf}`

### G5 等 epoch 与等 GPU hours 对比

同时报告相同 epoch 设置下的结果，以及按 GPU-hour 归一化后的对比。

```text
RobustGain = R_lite(defense) - R_lite(standard)
Efficiency = RobustGain / GPU_hours
```

## 脚本映射

训练：

```bash
python scripts/train.py --model preact_resnet18 --defense fixed_mixed_at --seed 0
```

单攻击评测：

```bash
python scripts/evaluate_attack.py --checkpoint checkpoints/.../best.pt --model preact_resnet18 --attack pgd20
```

AA-Lite：

```bash
python scripts/run_aalite.py --checkpoint checkpoints/.../best.pt --model preact_resnet18
```

AutoAttack 子集校验：

```bash
python scripts/run_autoattack_subset.py --checkpoint checkpoints/.../best.pt --model preact_resnet18 --subset-size 1000
```

梯度遮蔽诊断：

```bash
python scripts/run_diagnostics.py --checkpoint checkpoints/.../best.pt --model preact_resnet18 --max-samples 1000
```

结果聚合与出图：

```bash
python scripts/aggregate_results.py --input results/raw --output results/tables
python scripts/make_figures.py --tables results/tables --output results/figures
```

正式 real 模式下应使用：

```bash
python scripts/aggregate_results.py --input results/real/raw --output results/real/tables
python scripts/make_figures.py --tables results/real/tables --output results/real/figures
```

## 论文表格与图对应关系

- 表 1：模型 clean accuracy、参数量、训练时间，来源于 `main_robustness.csv`。
- 表 2：攻击鲁棒性矩阵，来源于 `main_robustness.csv`。
- 表 3：等 epoch 与等 GPU-hour 对比，来源于 `budget_comparison.csv`。
- 表 4：梯度遮蔽诊断标志，来源于 `gradient_masking_diagnostics.csv`。
- 图 2：结构鲁棒性热力图。
- 图 4：PGD-only vs AA-Lite 以及 `Gap_over`。
- 图 5：clean accuracy、`R_lite` 与 GPU-hour 权衡。
- 图 6：epsilon 单调性曲线。

## 执行策略

先运行 smoke 链路，例如使用 `--epochs 1 --max-samples 256`，确认 JSON、CSV 和至少一张图能够成功落盘。只有 smoke 链路完整通过后，才启动完整主实验矩阵。

正式论文结果必须来自 real 模式，不得混入 `fake_cifar10` 或 smoke/fake 输出。
