# 正式实验运行指南

完整可复制命令清单见：`docs/REAL_RUN_COMMANDS.md`。

## 从 Smoke 切换到 Real

- 正式 CIFAR-10 实验使用：`--dataset cifar10 --mode real --download`。
- 只聚合 real 输出：`python scripts/aggregate_results.py --input results/real/raw --output results/real/tables`。
- 论文表格不要使用 `--allow-smoke`。该参数只用于调试或检查 smoke/fake 结果，不应用于正式结果。
- 正式实验配置文件：
  - `configs/experiments/g1_structure_real.yaml`
  - `configs/experiments/g2_defense_real.yaml`
  - `configs/experiments/g3_eval_protocol_real.yaml`
  - `configs/experiments/g3_traps_real.yaml`
  - `configs/experiments/g4_mixed_ablation_real.yaml`
  - `configs/experiments/g5_budget_real.yaml`
- 推荐正式训练默认值：`epochs=80`，显存允许时 `batch_size=128`，`eps=8/255`，`pgd_at train_attack.steps=7`，`pgd20 eval steps=20`，`apgd_ce/apgd_dlr steps=50`。
- AA-Lite 建议使用固定子集配置 `configs/eval/subsets.yaml` 中的 `aalite_2k`；PGD-20 全测试集评测可使用 `--subset-size 0`。

## 算力参考

- 下面只是参考量级：在常见单 GPU 上，`PreActResNet-18 + PGD-7-AT + 80 epochs` 通常是“数小时”级别，不是几分钟级别。
- 在中端消费级 GPU 上，实际耗时可能从数小时到半天不等，主要取决于 batch size、数据加载速度、AMP 是否开启，以及 APGD 验证频率。
- 所有 latency、wall-clock、GPU-hour 数值都必须等 real 运行后再回填，不能使用 smoke/fake 结果替代。

## 推荐执行顺序

1. G1 real standard baselines：训练并评测 SmallCNN、ResNet-18、MobileNetV2，攻击包括 FGSM、PGD-20、APGD-CE-50、APGD-DLR-50。
2. G2 real defense matrix：在 PreActResNet-18 上比较 Standard、FGSM-AT、PGD-AT、Fixed Mixed-AT。
3. G3 protocol checks：检查 honest models、Trap-A、Trap-B、AA-Lite、Square subset，以及可选的 AutoAttack subset calibration。
4. G4 Fixed Mixed-AT ablation：在 G2 checkpoint 稳定后再做 Fixed Mixed-AT 组件消融。
5. G5 equal epoch / equal GPU-hour：每个对比组尽量在同一连续 session 中完成，保持 GPU、batch size、AMP、session_id 等字段一致。

## 本指南不覆盖的内容

- 完整 Budget-aware Scheduler 实现。
- MI-FGSM 迁移攻击。
- ViT 实验。
- VGG-16 实验。
- 将完整 AutoAttack 纳入主实验矩阵。

## 重要提醒

`results/smoke/raw/` 和 `results/smoke/tables/` 只用于工程链路验证，不可写入论文结果。正式论文只使用 `results/real/raw/`、`results/real/tables/` 和 `results/real/figures/` 下由真实 CIFAR-10 运行产生的结果。
