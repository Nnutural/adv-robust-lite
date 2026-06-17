# Codex 补丁实施提示词（C1-C9，按序执行）

> **直接把下面 `--- BEGIN PROMPT ---` 与 `--- END PROMPT ---` 之间的内容整段贴给 codex（或交给一个新开的 Claude/Codex agent）。Prompt 已经包含所有它需要的上下文与硬约束；不要再额外解释。**

---

--- BEGIN PROMPT ---

你是一名 AI 安全方向的资深 PyTorch 工程师，工作目录是 `adv-robust-lite/`。你的任务是按 `docs/patch_plan_v2.md` 把仓库从 P6 smoke 状态推进到满足 `Exp/plan-v2.md` C1-C9 的可论文运行状态。

## 必读文档（按顺序）

1. `Exp/plan-v2.md`（权威规格，冲突以此为准）
2. `Exp/plan-v1.md`（背景）
3. `Workout/GOAL/research_context.md`（RQ1-RQ4、AA-Lite、NormalizeWrapper 边界）
4. `adv-robust-lite/docs/audit_v2.md`（差距表，文件级现状定位）
5. `adv-robust-lite/docs/patch_plan_v2.md`（补丁规格，每个 C 的文件级改动清单、最小 diff 思路、验收断言、对现有 smoke 测试的影响）

读完后**不要重复总结**，直接进入实施。

## 实施顺序与"一 C 一组最小提交"节奏（不得违反）

按 **C1 → C2 → C3 → C4 → C5 → C6 → C7 → C8 → C9** 严格分组，**禁止跨 C 揉合改动**。每一组的执行循环：

1. 只改本 C 涉及的文件（路径见 `patch_plan_v2.md` 表格）。
2. 在 `tests/` 新增本 C 对应的 unit test（命名 `test_c{n}_*.py`）。
3. 跑 `python -m pytest tests -q`，所有既有测试 + 新增测试必须全 pass（torch 缺则 skipped 计为通过）。
4. 跑一次极小 fake smoke 链路（参考 `Workout/P6/command_log.md` 的命令，但**保持 fake 模式**），确认 `train → evaluate_attack → run_aalite → aggregate → make_figures` 不崩。
5. 在 `adv-robust-lite/docs/patch_reports/c{n}.md` 写一段不超过 30 行的报告：「改了什么 / 测试结果 / 是否满足 plan-v2 §11 对应 sanity gate」。
6. （可选）`git commit`，commit message 形如 `C{n}: <short summary>`。

**不要把多个 C 揉进一个大改动。** 若某个 C 的子项被自然分成两步（例如 C9 既有数据隔离又有目录迁移），允许在同一 C 下分 commit，但仍要 1-2 个 commit 即可。

## 硬约束（来自 plan-v2 §0 + §12 + 用户原则）

1. **不破坏最小闭环**：CIFAR-10 + SmallCNN/ResNet-18/MobileNetV2/PreActResNet-18 + FGSM/PGD-20/APGD-CE/APGD-DLR + Standard/FGSM-AT/PGD-AT/Fixed Mixed-AT 必须**始终可跑**。任何一步若让既有 6 个 smoke test 失败一项，必须立刻回滚本步。
2. **保留 `[0,1]` 像素空间攻击 + `NormalizeWrapper`**：扰动施加在归一化前，`clamp(0,1)` 在 NormalizeWrapper 外。不把 normalize 暴露给攻击库。
3. **smoke ≠ real**：fake_cifar10 数据**不得污染** real 聚合。C9 实施后 `results/real/raw/` 目录禁见 `dataset_name=="fake_cifar10"` 或 `mode=="smoke"` 的 JSON，聚合默认拒绝。
4. **不伪造指标**：所有数值来自真实运行并自动落盘。缺失（如 torchattacks/autoattack 未安装）→ `status="skipped"` + `error="..."`，**不许填 0、不许从其它攻击复制**。
5. **不擅自扩大范围**：本次任务不做 ViT、不做多数据集、不把完整 AA 放进主矩阵、不实现 Budget-aware Scheduler 完整增强；MI-FGSM/Square/VGG 仅在 C6（Square）/C1（Trap-B 配 Square）路径下出现，不进 G1/G2/G4 主矩阵。
6. **配置向后兼容**：所有新增 yaml 字段提供合理默认；旧 config 不传新字段时**走旧路径**且打一行 warning log，**不静默改变行为**。
7. **EOT 与"无 EOT 复现"必须显式标记**：对随机化对象（Trap-B）的白盒评测默认 `eot_samples >= 10`；用于"复现白盒被骗"的展示性评测必须输出 `eot_samples=0, eot_disabled_for_demo=true`，且**不得**与正式评测混在同一聚合行。
8. **不做 git push、不做 force operation、不动 git config**。所有 commit 走默认签名设置（不要 `--no-verify`、不要 `--amend` 已有 commit）。

## 关键工程纪律

- **测试驱动**：每个 C 必须先写或更新 `tests/test_c{n}_*.py`，再写实现；测试覆盖 `patch_plan_v2.md` 中"验收断言"的关键断言。
- **不主动修无关代码**：除非某行代码直接阻碍当前 C 的实施。`docs/audit_v2.md §10` 列的小问题，仅在对应 C 经过时顺手修（例如 `run_diagnostics.py` 漏传 `--dataset` 顺手在 C9 修）。
- **不写多余注释**：不解释代码做什么；只在出现 plan-v2 / paper 级别的不变式时加一行 why 注释（例如 "EOT required by plan-v2 §2.3 for randomized objects"）。
- **不创建多余 markdown**：除 `docs/patch_reports/c{n}.md` 与最终 `docs/ready_for_real_run.md` 之外，不要新建 docs。
- **保留既有命名与 CLI flag**：`patch_plan_v2.md` 已列出所有新增 flag；既有 flag（`--subset-size`、`--max-eval-batches`、`--steps` 等）一律保留向后兼容。
- **PowerShell 环境注意**：本仓库可能在 Windows + PowerShell 5.1 下被运行，命令中**不要**用 `&&` 与 `2>&1`；用 `; if ($?) { ... }` 或换行分开。

## 必跑的最小 smoke 命令（每个 C 后跑一次，**全程 fake 模式**）

```powershell
cd adv-robust-lite
python -m pytest tests -q
python scripts/train.py --model smallcnn --defense standard --epochs 1 `
  --dataset fake_cifar10 --mode smoke `
  --train-subset-size 128 --val-subset-size 32 `
  --max-train-batches 2 --max-eval-batches 1 --device cpu
python scripts/evaluate_attack.py --model smallcnn `
  --checkpoint checkpoints/smallcnn_standard_seed0/best.pt `
  --attack fgsm --dataset fake_cifar10 --mode smoke `
  --subset-size 32 --max-eval-batches 1 --device cpu
python scripts/evaluate_attack.py --model smallcnn `
  --checkpoint checkpoints/smallcnn_standard_seed0/best.pt `
  --attack pgd20 --dataset fake_cifar10 --mode smoke `
  --subset-size 16 --max-eval-batches 1 --steps 2 --device cpu
python scripts/run_aalite.py --model smallcnn `
  --checkpoint checkpoints/smallcnn_standard_seed0/best.pt `
  --dataset fake_cifar10 --mode smoke `
  --subset-size 8 --max-eval-batches 1 --steps 2 --device cpu
python scripts/aggregate_results.py --input results/smoke/raw --output results/smoke/tables --allow-smoke
```

> 说明：`--mode smoke` 与 `--allow-smoke` 是 C9 引入的；在 C1-C8 阶段如果还没实现，请回落到现有命令（无 `--mode`，输出走 `results/raw/`）。C9 完成后，主流命令永久切换为带 `--mode` 形式。

## 每个 C 完成的"完成定义"

- **C1**：极小 smoke 下，standard SmallCNN + `LogitScaleWrapper(scale=50)` 的 `Gap_over` 显著为正（>0.20），证明 RQ3 可证；`tests/test_c1_traps.py` 全 pass。
- **C2**：FGSM-AT/PGD-AT smoke 下 `metrics.json.best_criterion == "robust_val"`；standard 下 `== "clean_val"`；`tests/test_c2_best_robust_val.py` pass。
- **C3**：`pgd_at.yaml` 通过 `--defense-config` 加载后，trainer 实际使用 `train_attack.steps=7`，评测 `pgd20.steps=20`；断言 `train < eval`；`tests/test_c3_train_eval_split.py` pass。
- **C4**：FGSM-AT 默认走 `fgsm_rs`，`metrics.json` 含 `co_detected, co_epoch, co_pgd7_history`；`tests/test_c4_fgsm_rs_co.py` pass。
- **C5**：所有攻击 JSON 含 `eval_subset_id`；aalite 的 `gap_over` 在 ID 不一致时 = `None` 并写 `gap_over_error`；`tests/test_c5_subset_id.py` pass。
- **C6**：APGD 默认 steps=50；Square 加入 factory（缺包 skipped）；`tests/test_c6_apgd_steps_square.py` pass。
- **C7**：README + aalite payload 显式声明 `r_lite_scope=whitebox`。
- **C8**：metrics.json 与 budget_comparison.csv 都含 `gpu_name/batch_size/amp/session_id/max_wall_seconds`；`tests/test_c8_budget_columns.py` pass。
- **C9**：`mode=real` 拒绝 fake；`results/{mode}/raw/` 分离；聚合默认拒绝 smoke；既有 P6 fake JSON 已 git mv 到 `results/smoke/raw/`；`tests/test_c9_mode_isolation.py` pass。

## 最后一步：`docs/ready_for_real_run.md`

C9 完成且全部 sanity gate 通过后，写 `adv-robust-lite/docs/ready_for_real_run.md`，列出：

1. 从 smoke 切到 real 需要改的 config 项（`--dataset cifar10 --mode real --download`、各 `g*_real.yaml` 的 epoch/batch/eps 推荐值）。
2. 预计算力估算（按 PreActResNet-18 + PGD-7-AT + 80 epoch 在常见 GPU 上的 wall-clock 范围给一个区间，不要凭空精确数字；标注"参考量级"）。
3. 推荐运行顺序（P7 / P8 等价的最小可发表实验序列）。
4. 列出**未实施的增强项**（Budget-aware Scheduler、MI-FGSM、ViT、VGG-16），明确标注"超出本次范围"。

不要在 ready_for_real_run.md 里假装跑出真实数字，所有 latency/wall-clock 都写"待 real 运行后回填"。

## 禁止事项（再强调）

- 不要重写 NormalizeWrapper 或 [0,1] 攻击设计。
- 不要在 Standard 训练上启用 robust-val（除非 yaml 显式打开）。
- 不要把 PGD-only 改名或合并到 R_lite。
- 不要把 Trap-A/Trap-B 数据写进 G1/G2/G4 主矩阵；它们只在 G3 与诊断里出现。
- 不要在 EOT 缺失时把 Trap-B 的白盒虚高数字呈现为"正式结果"。
- 不要把 fake 数据的结果聚合进 `results/real/`。
- 不要在 commit message 中写"按 plan-v1"或"按旧规格"——一切以 plan-v2 为准。

--- END PROMPT ---
