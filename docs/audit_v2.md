# audit_v2 — 当前实现对照 plan-v2 的差距表

> Scope：审计 `adv-robust-lite/` 截至 P6 的实现，与 `Exp/plan-v2.md` 的 C1-C9 逐条对照。本文件**只识别差距**，不给出代码改动方案；改动方案见 `docs/patch_plan_v2.md`。
>
> 关键约束（来自 plan-v2 §0 / §12）：保留最小闭环、目录结构、configs 驱动、`NormalizeWrapper` + `[0,1]` 攻击设计、AA-Lite/`R_lite`/`Gap_over` 形式化定义；冲突时以 plan-v2 为准。

---

## 0. 差距汇总（按严重度 + plan-v2 编号）

| 编号 | 严重度 | 目标行为（plan-v2） | 现状（文件/行） | 差距 | 风险 | 改动范围估计 |
|---|---|---|---|---|---|---|
| C1 | 🔴 科学性 | Trap-A logit 缩放 wrapper + Trap-B 输入随机化 wrapper + EOT 白盒 | 无（`src/models/factory.py:13-37` 只有 4 个诚实模型，`src/attacks/factory.py:57-119` 的 PGD 无 EOT） | 完全缺失：无 wrapper、无 EOT、无对应 defense/config、无 G3+ 诊断数据生成路径 | RQ3 无法证明（诚实 AT 的 `Gap_over ≈ 0`，做不出 AA-Lite 比 PGD-only 强的证据）；Trap-B 的「黑盒>白盒」诊断没东西可触发 | 新增 5 文件 + 改 2 文件：`src/models/trap_wrappers.py`（新）、`src/attacks/eot.py`（新）、`configs/defenses/trap_logit.yaml`+`trap_random.yaml`（新）、`configs/experiments/g3_traps_smoke.yaml`+`g3_traps_real.yaml`（新）、`src/models/factory.py`（注册）、`scripts/run_aalite.py`（透传 EOT 旗标） |
| C2 | 🟠 正确性 | `best.pt` 按 robust val (PGD-7 或 AA-Lite-small) 选；`metrics.json` 写 `best_criterion/best_metric/best_metric_value/best_epoch` | `src/training/trainer.py:170,199-201` 按 `val_acc`（clean）选 best；`metrics.json` 不含 best_criterion 字段 | best 选错维度；无字段可断言 | 鲁棒过拟合点位被 clean 峰值掩盖；G2/G4 的 R_lite 偏低且不可解释 | 改 1 文件 + 加 1 段断言：`src/training/trainer.py`（新增 robust-val 评估方法 + best 路径分流 + metrics 写盘）；测试 `tests/test_training_smoke.py` 补 `best_criterion=="robust_val"` 断言 |
| C3 | 🟠 正确性 | configs 显式分 `train_attack` 与 `eval.attacks`；训练 PGD-7，评测 PGD-20/APGD-50；断言 `train_attack.steps < eval pgd20 steps` | `configs/defenses/pgd_at.yaml:3-8` 已有 `train_attack` 字段，但 `trainer.py:58-75` **完全不读它**，仍走 `TrainConfig.pgd_steps=7` 的 CLI 默认；`scripts/evaluate_attack.py:18,33` `--steps 20`、`apgd_*.yaml:4` `steps: 20` 与 plan-v2 的 50 不一致；没有任何 sanity gate 校验两者关系 | 训练攻击与评测攻击没解耦；APGD 步数与 plan-v2 不一致；可能把评测攻击误配进训练 | trainer 改 + configs 改 + 所有 defense yaml 加 `train_attack` 段；运行时断言 train<eval | 改 4 文件 + 加 1 测试：`src/training/trainer.py`（解析 `train_attack` 段并覆盖 `TrainConfig`）、`configs/defenses/*.yaml`（统一格式 + `eval:` 段）、`configs/attacks/apgd_*.yaml`（`steps: 50`）、`scripts/train.py`（加载 defense yaml 时透传）、`tests/test_config.py`（新增 train<eval 断言） |
| C4 | 🟠 正确性 | FGSM-AT 改 FGSM-RS（random init + α≈10/255 单步），新增 CO 检测器（PGD-7 val 单轮跌幅>15pp 触发早停 + `co_detected/co_epoch` 字段） | `src/training/trainer.py:62`→`build_attack_config("fgsm")`→`src/attacks/factory.py:57-73` 的 `fgsm_attack` 无 random init、α=ε；`src/training/fgsm_at.py:1-8` 是壳子，没有 PGD-val 监控 | 朴素 FGSM-AT 会灾难性过拟合到 PGD≈0；现状没有 CO 检测、没有 `co_detected` 字段；FGSM-RS 完全缺失 | RQ2 中 FGSM-AT 数据无解释；论文里 CO 现象无法作为发现展示 | 改 2 文件 + 新增 1 段：`src/attacks/factory.py`（加 `fgsm_rs_attack(model, x, y, eps, alpha, random_start=True)`）、`src/training/trainer.py`（FGSM-AT 用 FGSM-RS + 每 N epoch 跑 PGD-7 val + CO 检测+早停字段）、配置 `configs/defenses/fgsm_at.yaml`（加 `train_attack: name: fgsm_rs, alpha: 10/255` + `co_check.steps: 7 / threshold: 0.15`） |
| C5 | 🟡 成本/可比 | PGD-20 全 10k；APGD/`R_lite`/`Gap_over` 在**同一固定 2,000 子集**；输出携带 `eval_subset_id`，聚合校验三攻击同 ID | `scripts/evaluate_attack.py:27,49-57`、`scripts/run_aalite.py:25,47-55` 各传 `--subset-size`，没有 `eval_subset_id` 字段；`src/attacks/runner.py:113-127`、`src/attacks/aalite.py:50-68` 输出 JSON 无 `eval_subset_id`；`scripts/aggregate_results.py:31-69` 不做同子集校验 | `Gap_over = pgd20_acc - r_lite` 可能跨不同子集；plan-v2 §11 sanity gate 无法满足 | RQ3 的 `Gap_over` 数据不可比；论文无法主张 PGD-only 与 R_lite 同口径 | 改 4 文件 + 加 1 字段：`src/datasets/cifar10.py`（新增 `build_eval_subset(name, size, seed)` 返回 `(loader, subset_id)`，subset_id = sha1(name|size|seed) 截断）、`src/attacks/runner.py`（metadata 透传 `eval_subset_id`）、`src/attacks/aalite.py`（同上）、`scripts/aggregate_results.py`（聚合时按 `exp_id` 分组，校验 `pgd20/apgd_ce/apgd_dlr` 的 `eval_subset_id` 相同，否则把 `gap_over=NA` 且 `gap_over_error="subset_id_mismatch"`） |
| C6 | 🟡 成本 | APGD steps 固定 50；Square `n_queries=2000`（subset=1000） | `configs/attacks/apgd_ce.yaml:4`、`apgd_dlr.yaml:4` 写 `steps: 20`；`scripts/evaluate_attack.py:19`、`scripts/run_aalite.py:18` `--steps 20` 默认；Square 整个未实现（`src/attacks/factory.py:151-179` 的 `AttackFactory.create` 没有 square 分支） | APGD 默认 20 与 plan-v2 不符；Square 缺；Trap-B 的黑盒诊断没有 Square 可用 | 评测协议参数与论文不一致；Trap-B 故事线缺少证据 | 改 2 文件 + 新增 1 段：`configs/attacks/apgd_*.yaml`（`steps: 50`）、CLI default 同步；`src/attacks/factory.py` 加 `square_attack`（依赖 torchattacks/autoattack，缺则 skipped 不伪造）、`configs/attacks/square.yaml`（新增 `n_queries: 2000`, `subset_size: 1000`） |
| C7 | 🟡 表述 | `R_lite` 是白盒-only；README 与 metrics 输出处显式声明 | `README.md:66-75` 没说明白盒-only；`src/attacks/aalite.py:13` 注释只列三攻击名；JSON 输出无 `r_lite_scope: whitebox` 字段 | 论文/复现者会误以为 `R_lite` 已覆盖黑盒；Trap-B 抓不到时易被误读 | README 论述脆弱；下游用户错误解释 | 改 2 文件：`README.md`（加 "AA-Lite 仅白盒；Square + 黑盒>白盒诊断单算" 段落）、`src/attacks/aalite.py`（payload 加 `"r_lite_scope": "whitebox", "blackbox_handled_separately": true`） |
| C8 | 🟡 公平 | 等 GPU 小时比较硬前提：同机/同 batch/同 AMP/同会话；`g5_budget_comparison.csv` 每行带 `gpu_name/batch/amp` | `src/training/trainer.py:204-220` 写 `metrics.json` 时只有 `device + 可选 gpu_name`，没有 `batch/amp` 字段一起记；`scripts/aggregate_results.py:116-145` 出的 `budget_comparison.csv` 没有 `gpu_name/batch/amp` 列也不做一致性校验 | 等 GPU 小时表无法证明三方等价 | RQ4 等算力对比可信度低 | 改 2 文件：`src/training/trainer.py`（`metrics.json` 写 `gpu_name/batch_size/amp/session_id`）、`scripts/aggregate_results.py`（`budget_comparison.csv` 加这三列并断言同 comparison 内三者一致，不一致打标） |
| C9 | 🟡 隔离 | `configs/experiments/*_smoke.yaml` 与 `*_real.yaml` 分离；datasets 加 `use_fake_data` 开关与 real 模式 fake 禁用断言；结果 `results/smoke/` 与 `results/real/` 分目录；聚合默认只读 real；fake 标记进入 real 必须报错 | `configs/experiments/` 当前只有 `g1_structure/g2_defense/g3_eval_protocol/g4_mixed_ablation/g5_budget.yaml` 共 5 文件，**没有 smoke/real 后缀**；`src/datasets/cifar10.py:38-65` 通过 `dataset_name="fake_cifar10"` 切换但没有 real-mode 禁用断言；`results/raw/` 既装真实结果也装 fake，无目录隔离；`scripts/aggregate_results.py:151-153` 默认 `--input results/raw` 不区分 | 论文表可能不小心混入 P6 fake 数据 | 论文数据完整性事故 | 改 4 文件 + 新增多个配置：`src/datasets/cifar10.py`（加 `mode: real / smoke` 参数 + real 模式 fake 即抛 `RuntimeError`）、`scripts/train.py`+所有 eval 脚本（默认输出到 `results/{mode}/raw/...`，metadata 写 `mode: smoke|real`）、`scripts/aggregate_results.py`（默认 `--input results/real/raw`，遇 `mode: smoke` 或 `dataset_name: fake_cifar10` 行报错）、新增 `configs/experiments/g*_smoke.yaml` 与 `g*_real.yaml` 全套 |

---

## 1. C1 — Trap-A / Trap-B / EOT（核心，关系 RQ3 成立）

### 现状定位

- `src/models/factory.py:13-37`：`build_model` 只支持 `smallcnn/resnet18/mobilenetv2/preact_resnet18` 四个诚实 backbone，且强制走 `NormalizeWrapper`。没有任何外层 logit-scale 或输入随机化 wrapper 的注册位。
- `src/models/normalize_wrapper.py:4-24`：`NormalizeWrapper` 只做 (x-mean)/std，没有给后续 wrapper 链留接口。
- `src/attacks/factory.py:76-119`：`pgd_attack` 是单次梯度路径，没有「同一 step 对随机化做多次采样平均」的 EOT 支持。
- `configs/defenses/*.yaml`：四个 yaml 文件，全部是诚实训练，没有 trap 对象配置。

### 与 plan-v2 §2 对照的差距

| plan-v2 §2 要求 | 现状 | 差距 |
|---|---|---|
| Trap-A：standard 模型外套 wrapper，推理时 logits *= T (T=50) | 完全没有 | 需新增 `LogitScaleWrapper`（forward 包 backbone，乘 T；可挂在 NormalizeWrapper 外）+ 一个 trap_logit defense（对应训练就是 standard，但加载时套上 wrapper） |
| Trap-B：standard 模型前加随机变换 wrapper（random resize+pad 或随机量化） | 完全没有 | 需新增 `InputRandomizationWrapper`（forward 时对 [0,1] 图像随机 padding/resize 或随机量化，每次 forward 重采样）+ 一个 trap_random defense |
| 对 Trap-B 的白盒评测必须支持 EOT (`eot_samples >= 10`) | 完全没有 | `pgd_attack`/`apgd_*` 都要支持「对随机化对象多采样平均梯度」；需要新增 `eot_pgd` 或在 `AttackConfig` 上加 `eot_samples` 字段 |
| 「无 EOT 跑白盒」也保留用于复现「被骗」 | 默认就是无 EOT，但需要**显式标记** | runner 输出 JSON 要写 `eot_samples: 0` 与 `eot_samples_used` 区分 |

### 验收难度

- 极小 smoke 即可证明 Trap-A 的 `Gap_over` 显著为正：standard 训完任何模型，套 LogitScaleWrapper(T=50)，跑 PGD-CE（梯度因 softmax 饱和接近 0，robust 虚高）vs APGD-DLR（对 logit 整体缩放不变，应戳穿）。
- Trap-B 需要 Square 子集才能完成「黑盒>白盒」对比，因此**C1 验收依赖 C6 的 Square**。如果 Square 未到位，C1 仅能验收 Trap-A 故事线。

---

## 2. C2 — best 选择改 robust val

### 现状定位

- `src/training/trainer.py:170-201`：

  ```text
  best_val_acc = -1.0
  best_path = self.run_dir / "best.pt"
  for epoch in ...:
      val_stats = self.evaluate_clean()        # ← clean val
      ...
      if val_stats["val_acc"] > best_val_acc:  # ← 用 clean 选 best
          best_val_acc = val_stats["val_acc"]
          save_checkpoint(best_path, checkpoint)
  ```

- `src/training/trainer.py:204-222`：`metrics.json` 字段：

  ```text
  exp_id, model, defense, seed, clean_acc(==best_val_acc), train_time_sec,
  train_time_gpu_hours, params, checkpoint_path, device, max_train_batches, max_eval_batches, [gpu_name]
  ```

  没有 `best_criterion / best_metric / best_metric_value / best_epoch`。

### 与 plan-v2 §3 对照的差距

| plan-v2 §3 要求 | 现状 | 差距 |
|---|---|---|
| 每 `eval_every` 在固定 val 子集算 robust acc（PGD-7 或 AA-Lite-small） | 没有 robust-val 评估方法 | 需新增 `evaluate_robust(steps=7)`；val_loader 固定一份子集（约 500-1000 张），用于轮内监控 |
| `best.pt` 取 robust-val 峰值；clean-best 另存为 `best_clean.pt` | 只有一份 `best.pt`（clean） | 需要分流保存 |
| `metrics.json` 新增 best_criterion/best_metric/best_metric_value/best_epoch | 全缺 | 改写字段 |
| 断言 `best_criterion == "robust_val"` | 无测试 | 新增断言 |
| Standard 模型可保留 clean-val（plan-v2 没强制对 standard 跑 robust val） | — | 实现要分支：仅 AT 类 defense 默认 robust val；standard 保留 clean-val |

### 风险点

- robust val 即使是 PGD-7、512 张子集，单 epoch 也会增加显著时间（CIFAR-10 PreActResNet-18 大约 +5-15s/epoch）。需在 config 提供 `eval_every` 控制（默认 1 或 5）。
- 极端低算力 smoke（max_train_batches=2）下 robust val 可能没有意义，需要在 smoke 模式下允许回落 clean-val 但显式打标记。

---

## 3. C3 — train_attack ≠ eval_attack 解耦

### 现状定位

- `configs/defenses/pgd_at.yaml:1-8`：写了 `train_attack:` 段，但实际上：

  ```text
  src/training/trainer.py:58-75   _attack_batch() 只看 TrainConfig.pgd_steps / pgd_alpha
  scripts/train.py:62-76          构造 TrainConfig 时根本不读 configs/defenses/pgd_at.yaml
  ```

  也就是说 `configs/defenses/pgd_at.yaml` 现在是**装饰性配置**，对 trainer 无效。

- 评测：`scripts/evaluate_attack.py:18-19` `--steps 20` 默认；`scripts/run_aalite.py:18` `--steps 20` 默认；`configs/attacks/apgd_*.yaml:4` 都是 20。
- 没有 sanity 检查训练步数 < 评测步数。

### 与 plan-v2 §4 对照的差距

| plan-v2 §4 要求 | 现状 | 差距 |
|---|---|---|
| configs 分 `train_attack`/`eval.attacks` | `train_attack` 部分有但 trainer 不读；`eval` 段全缺 | trainer 必须实际消费 defense yaml；eval 段需新增 |
| 训练内层 PGD-7（α=2/255, random start） | trainer 默认 `pgd_steps=7, pgd_alpha=2/255, random_start=True` ✓ 数值对，但路径来自 CLI 不来自 config | 走通配置链 |
| 评测 PGD-20 + APGD-CE-50 + APGD-DLR-50 | PGD-20 ✓；APGD-50 ✗（当前 20） | C6 一起改 |
| 训练日志记录实际训练攻击步数 | trainer 不记 | 加进 `metrics.json` 的 `train_attack_config` |
| 断言 `train_attack.steps < eval.pgd20.steps` | 无 | 新增 unit test |

---

## 4. C4 — FGSM-AT 改 FGSM-RS + CO 检测

### 现状定位

- `src/attacks/factory.py:57-73` `fgsm_attack`：

  ```text
  adv = images + eps * grad.sign()   # 无 random init
  ```

  α 隐含为 eps；无 random start。

- `src/training/trainer.py:60-75 / 86-98`：FGSM-AT 路径直接走 `build_attack_config("fgsm", eps=self.config.eps)` → `fgsm_attack`，所以 FGSM-AT 实际是**朴素单步**，符合"灾难性过拟合"的高危条件。
- 无任何 PGD-7 val 监控、无 `co_detected` 字段。

### 与 plan-v2 §5 对照的差距

| plan-v2 §5 要求 | 现状 | 差距 |
|---|---|---|
| FGSM-AT 默认改为 FGSM-RS（random init + α≈10/255 单步） | 当前是朴素 FGSM，α=ε | 新增 `fgsm_rs_attack(model, x, y, eps, alpha=10/255, random_start=True)`；FGSM-AT 默认改用它 |
| CO 检测器：训练中按 `eval_every` 在 val 跑 PGD-7，**单轮**相对暴跌 >15pp 触发早停 + 落盘标记 | 完全缺 | 需在 trainer 内加 `co_check`（与 C2 robust-val 可复用同一 evaluator）+ 早停逻辑 + 写 `co_detected / co_epoch` 字段 |
| metrics.json 含 `co_detected: bool, co_epoch: int` | 缺 | 新增字段 + 默认 False/None |
| 启用 FGSM-RS 时 PGD-val 不应坍塌到 ≈0 | 缺验证 | 极小 smoke 跑 1-2 epoch 不一定能验，主要是大 epoch 实验时观察 |

---

## 5. C5 — 评测子集 `eval_subset_id` 与同子集校验

### 现状定位

- `src/datasets/cifar10.py:184-222` `build_cifar10_test_loader`：通过 `seed + subset_size` 重新生成 `randperm` 子集索引；外面不传 `subset_indices_path` 时**每次调用都重生**但因为同 seed 同 size 同种子序列，子集索引在数学上一致。但**没有暴露子集指纹**给攻击 runner，也没写入 JSON。
- `src/attacks/runner.py:113-127` 与 `src/attacks/aalite.py:50-68`：输出 JSON 字段中无 `eval_subset_id`。
- `src/attacks/aalite.py:64-67`：`Gap_over = pgd20_acc - r_lite`，此处 PGD-20 是 AA-Lite 内自己跑的 PGD-20，二者**同子集**——但 plan-v2 §6 的语义实际是**"PGD-only 全集 10k vs R_lite 2k 子集"**，所以这里有两种解释：
  - 解释 A（保守）：`Gap_over` 内部 PGD-only 就是 AA-Lite 里那次 PGD-20 → 同子集 ✓，但全集 10k 上的 PGD-only 也要保留为独立字段以呈现「PGD-only 高估」。
  - 解释 B（plan-v2 §6 原意）：`Gap_over` 的 PGD-only 与 `R_lite` 必须同子集，**且都是 2000 子集**。

  无论哪种解释，当前都缺 `eval_subset_id` 字段无法证明。
- `scripts/aggregate_results.py:31-69`：聚合时根本不看 `eval_subset_id`。

### 与 plan-v2 §6 对照的差距

| plan-v2 §6 要求 | 现状 | 差距 |
|---|---|---|
| PGD-20 全 10k 评测 | 当前 PGD-20 子集任意可变 | 加 `--full-test` 选项 + 默认整 test set |
| APGD-CE/DLR/`R_lite`/`Gap_over` 固定 2000 子集 | 子集大小靠 CLI 传入，没有强制 2000 | 配置默认 2000；攻击/聚合都按子集 ID 校验 |
| `R_lite/Gap_over` 携带 `eval_subset_id` | 全缺 | 新增 |
| 聚合校验三攻击同 `eval_subset_id` | 不做 | 新增校验，不一致→`gap_over=NA` 且写 `error` |

---

## 6. C6 — APGD steps=50 + Square

### 现状定位

- `configs/attacks/apgd_ce.yaml:4` `steps: 20`、`apgd_dlr.yaml:4` `steps: 20`、`pgd20.yaml:4` `steps: 20`、CLI `--steps 20` 默认。
- `src/attacks/factory.py:151-179`：`AttackFactory.create` 仅 fgsm/pgd/apgd_ce/apgd_dlr 四种分支，无 Square。
- Square 完全缺。

### 与 plan-v2 §7 对照的差距

| plan-v2 §7 要求 | 现状 | 差距 |
|---|---|---|
| APGD steps = 50（不写 20-50） | 20 | configs 与 CLI default 全改 50 |
| APGD restarts=1 | 已 1 ✓ | — |
| Square 1000 子集，`n_queries=2000` | 全缺 | 新增 `src/attacks/square.py` 或 `factory.py` 分支（torchattacks.Square 或 autoattack 的 Square），配 `configs/attacks/square.yaml` |
| Trap-B 白盒/EOT `eot_samples >= 10` | 缺 | 与 C1 一起做 |
| `R_lite` 文档处显式注明白盒-only | 缺 | C7 一起做 |

---

## 7. C7 — R_lite 白盒-only 显式声明

### 现状定位

- `README.md:66-75`：

  ```text
  R_lite = min(Acc_PGD20, Acc_APGD_CE, Acc_APGD_DLR)
  Gap_over = Acc_PGD20 - R_lite
  ```

  没有说"白盒-only"、没有解释"Square + 黑盒>白盒 诊断单独评估"。
- `src/attacks/aalite.py:50-72`：payload 没有 `r_lite_scope`/`blackbox_handled_separately` 字段。

### 差距

仅文档与一个 metadata 字段，**改动最小**，可与 C1/C6 同补丁合并提交。

---

## 8. C8 — 等 GPU 小时公平前提

### 现状定位

- `src/training/trainer.py:204-220` 写 `metrics.json` 时缺 `batch_size` 和 `amp`；只在 cuda 才写 `gpu_name`，CPU 路径就空着。
- `scripts/aggregate_results.py:116-145` 生成 `budget_comparison.csv` 时列只有：`defense, epoch_budget, gpu_hour_budget, clean_acc, r_lite, clean_drop, robust_gain, gain_per_gpu_hour`，无 `gpu_name/batch/amp`，也不做一致性校验。

### 差距

| plan-v2 §8 要求 | 现状 | 差距 |
|---|---|---|
| 每行 `g5_budget_comparison.csv` 带 `gpu_name/batch/amp` | 三列全缺 | 加列 |
| 同一比较内三者一致 | 不校验 | 加校验，不一致→标 `equal_budget_invalid=true` |
| 以 wall-clock 预算 + 周期 checkpoint 截断训练 | trainer 不支持时间截断 | 新增 `--max-wall-seconds` CLI + trainer 内每 batch/epoch 末判时间 |

---

## 9. C9 — smoke / real 物理隔离

### 现状定位

- `configs/experiments/` 当前 5 个 yaml 文件，**无** `_smoke` / `_real` 后缀；都是开发期 stub（如 `g3_eval_protocol.yaml` 只列 protocols 名）。
- `src/datasets/cifar10.py:38-65`：`dataset_name="fake_cifar10"` 自由切换，**无 real-mode 禁用 fake 的断言**。
- `scripts/train.py:24`、`scripts/evaluate_attack.py:24`、`scripts/run_aalite.py:23` 都接收 `--dataset {cifar10, fake_cifar10}` 默认 cifar10——但运行时没有 `mode` 概念。
- 结果目录：`results/raw/` 当前同时承载 P6 的 fake smoke 与未来 real 结果；P6 写出的 fake JSON（`smallcnn_standard_seed0_*.json` 等）**没有 `dataset` 或 `mode` 字段**保护。
- `scripts/aggregate_results.py:18-28`：`rglob("*.json")` 全收，对 dataset/mode 不过滤。

### 差距

| plan-v2 §9 要求 | 现状 | 差距 |
|---|---|---|
| `configs/experiments/*_smoke.yaml` 与 `*_real.yaml` 分离 | 全缺 | 新增 10 个 yaml（5 实验 × 2 模式） |
| datasets 加 `use_fake_data` 开关 + real 禁用 fake | 现状 dataset_name 是 free toggle | 加 `mode: real / smoke` 参数；real 见到 fake → `RuntimeError` |
| 结果 `results/smoke/` 与 `results/real/` 分目录 | 全混合 | 所有脚本根据 `mode` 决定输出根；加 `results/{mode}/raw/...` 子树 |
| 聚合默认只读 real，遇 fake 标记报错 | 不做 | 默认 `--input results/real/raw`；遇 `dataset==fake_cifar10` 或 `mode==smoke` 的 JSON 抛错并列文件 |
| P6 报告处醒目标注 smoke-only | `Workout/P6/stage_report.md` 已有，但 `results/reports/01_audit_report.md` 未审 | 在 README 加 banner 段 |

---

## 10. 不在 C1-C9 内但顺带发现的小问题（仅记录，不必这次修）

- `src/utils/checkpoint.py:7-35`：`save_checkpoint` 接受 dict 或 model 两种 payload，存在一定调用歧义，但当前 trainer 始终传 dict，先不动。
- `src/attacks/runner.py:74-79`：`adv_images` 范围 assert 用 `1e-6` 容差合理，与 plan-v2 没冲突。
- `tests/test_training_smoke.py:11-40`：当前只验 `clean_acc>=0` 与文件存在，需要在 C2 实施后扩成断言 `best_criterion`。
- `scripts/run_diagnostics.py:56-63`：构造 loader 时**漏传 `dataset_name`**，导致诊断脚本只能在真 CIFAR-10 上跑（fake_cifar10 模式下会直接尝试下载真数据）；这是一个独立小 bug，建议在 C9 同补丁里顺带补 `--dataset` 参数。

---

## 11. 修复优先级映射回 plan-v2

按 plan-v2 §1 严重度顺序：

1. **C1（🔴）** — Trap-A/Trap-B + EOT：决定 RQ3 能否成立，最先做。
2. **C2 / C3 / C4（🟠）** — best 选择 / train≠eval / FGSM-RS+CO：决定主矩阵 G2/G4 的数据是否可信。
3. **C5（🟡）** — eval_subset_id：决定 RQ3 数据可比性。
4. **C6（🟡）** — APGD-50 + Square：补齐协议参数，并喂给 C1 的 Trap-B 故事线。
5. **C7 / C8（🟡）** — R_lite scope 声明 / 等 GPU 小时三列：表述与公平。
6. **C9（🟡）** — smoke/real 隔离：保险栓，与 README banner 同时做，避免论文出事故。

每个 C 都对应 plan-v2 §11 sanity gate 中的一条断言，详见 `docs/patch_plan_v2.md`。
