# patch_plan_v2 — 文件级补丁计划

> 范围：把 `adv-robust-lite/` 从 P6 smoke 推进到满足 `Exp/plan-v2.md` C1-C9 的可论文运行状态。
>
> 原则（plan-v2 §0 + 用户硬约束）：
> 1. **不破坏最小闭环**：CIFAR-10 + 4 个 backbone + FGSM/PGD/APGD-CE/APGD-DLR + Standard/FGSM-AT/PGD-AT/Fixed Mixed-AT 必须一直可跑。
> 2. **保留 [0,1] + NormalizeWrapper**，不把 normalize 暴露给攻击库。
> 3. **smoke ≠ real**：fake 数据不得污染 real 聚合；real 聚合默认只读 `results/real/`。
> 4. **不伪造指标**：缺就标 N/A。
> 5. **配置向后兼容**：旧 config 不应静默改变行为。新字段提供合理默认；缺省走旧路径并打日志说明。
> 6. **随机化对象的白盒评测必须 EOT**；故意不带 EOT 用于"复现白盒被骗"时必须显式标记 `eot_samples=0, eot_disabled_for_demo=true`。
>
> 提交节奏：**一个 C 一组最小补丁**，禁止把多个 C 揉成一个大改动。每组完成后：
> - 跑相关 smoke test（已有 + 本组新增的断言）
> - 跑一次极小 fake smoke 确认链路不崩
> - 在 commit message 或 patch report 里写明「改了什么 / 测试结果 / 是否满足 plan-v2 对应 sanity gate」

---

## 0. 全局约定（先于 C1 实施）

### 0.1 新增 / 修改约定

- 所有新增字段在 `metrics.json` / 攻击 JSON 中**追加**，不删除既有字段。
- 配置文件新字段提供默认值（在 `configs/base.yaml` 或 trainer/runner 内）；旧 yaml 不需要改也能跑（走旧行为 + 一条 warning log）。
- 命名约定：
  - 文件级 ID：`subset_id = sha1(f"{name}|{size}|{seed}").hexdigest()[:12]`
  - mode：`real | smoke`（字符串）
  - best 字段：`best_criterion ∈ {robust_val, clean_val}`、`best_metric ∈ {pgd7_val_acc, val_acc}`
  - r_lite scope：`r_lite_scope = "whitebox"`、`blackbox_handled_separately=true`

### 0.2 测试约定

- 现有 `tests/` 6 个 smoke test 必须**全部继续 pass**。
- 每个 C 新增至少 1 个 unit test（torch-optional 用 `pytest.importorskip("torch")`）。
- 命名：`tests/test_c{n}_*.py`，例如 `tests/test_c1_traps.py`。

### 0.3 目录新增

```
adv-robust-lite/
├── configs/
│   ├── defenses/
│   │   ├── trap_logit.yaml          (新增, C1)
│   │   └── trap_random.yaml         (新增, C1)
│   ├── attacks/
│   │   └── square.yaml              (新增, C6)
│   ├── eval/
│   │   └── subsets.yaml             (新增, C5)
│   └── experiments/
│       ├── g1_structure_smoke.yaml  (新增, C9)
│       ├── g1_structure_real.yaml   (新增, C9)
│       ├── g2_defense_smoke.yaml    (新增, C9)
│       ├── g2_defense_real.yaml     (新增, C9)
│       ├── g3_eval_protocol_smoke.yaml  (新增, C9)
│       ├── g3_eval_protocol_real.yaml   (新增, C9)
│       ├── g3_traps_smoke.yaml      (新增, C1)
│       ├── g3_traps_real.yaml       (新增, C1)
│       ├── g4_mixed_ablation_smoke.yaml (新增, C9)
│       ├── g4_mixed_ablation_real.yaml  (新增, C9)
│       ├── g5_budget_smoke.yaml     (新增, C9)
│       └── g5_budget_real.yaml      (新增, C9)
├── src/
│   ├── attacks/
│   │   ├── eot.py                   (新增, C1)
│   │   ├── fgsm_rs.py               (新增, C4，或直接加在 factory.py)
│   │   └── square.py                (新增, C6)
│   └── models/
│       └── trap_wrappers.py         (新增, C1)
├── docs/
│   ├── audit_v2.md                  (已写, 第1步)
│   ├── patch_plan_v2.md             (本文件)
│   └── ready_for_real_run.md        (最后一步)
└── tests/
    ├── test_c1_traps.py             (新增)
    ├── test_c2_best_robust_val.py   (新增)
    ├── test_c3_train_eval_split.py  (新增)
    ├── test_c4_fgsm_rs_co.py        (新增)
    ├── test_c5_subset_id.py         (新增)
    ├── test_c6_apgd_steps_square.py (新增)
    ├── test_c8_budget_columns.py    (新增)
    └── test_c9_mode_isolation.py    (新增)
```

---

## C1 — Trap-A logit 缩放 + Trap-B 输入随机化 + EOT

### 文件级改动

| 操作 | 路径 | 说明 |
|---|---|---|
| **新增** | `src/models/trap_wrappers.py` | 两个 `nn.Module`：`LogitScaleWrapper(backbone, scale)` 与 `InputRandomizationWrapper(backbone, kind, pad, jitter)` |
| **新增** | `src/attacks/eot.py` | `eot_grad(model, x, y, loss_fn, eot_samples)` → 返回平均梯度；`pgd_attack_eot` 在每 step 内调用 |
| **修改** | `src/models/factory.py` | `build_model(...)` 增加 `wrappers: list[dict] | None = None` 参数：可在 NormalizeWrapper **外**再套 trap wrapper；保留默认 `wrappers=None`（旧行为不变） |
| **新增** | `configs/defenses/trap_logit.yaml` | `defense.name: standard`（不动训练）+ `eval.model_wrappers: [{kind: logit_scale, scale: 50.0}]` |
| **新增** | `configs/defenses/trap_random.yaml` | 同上 + `[{kind: input_random, pad: 4, kind_random: "pad_resize"}]` + `eval.eot_samples: 10` |
| **修改** | `src/attacks/factory.py` | `AttackConfig` 新增 `eot_samples: int = 0`；`pgd_attack` 接受 `eot_samples`；`>0` 时走 `eot.py` 路径 |
| **修改** | `src/attacks/runner.py` | metadata 输出 `eot_samples: int, eot_disabled_for_demo: bool` |
| **修改** | `src/attacks/aalite.py` | 同上；payload 写 `r_lite_scope: "whitebox"` |
| **修改** | `scripts/run_aalite.py` + `scripts/evaluate_attack.py` | 新增 `--eot-samples`、`--model-wrappers <yaml_path>` CLI 选项 |
| **新增** | `configs/experiments/g3_traps_smoke.yaml` + `_real.yaml` | 列出三组对比：诚实 standard vs trap_logit vs trap_random，分别跑 PGD-only / AA-Lite / Square |
| **新增** | `tests/test_c1_traps.py` | 验：(a) `LogitScaleWrapper(scale=50)` 套在 SmallCNN 上，PGD-CE 比 APGD-DLR 的 robust_acc 显著更高；(b) `InputRandomizationWrapper` forward 每次输出不同；(c) `eot_samples=4` 路径不抛错 |

### 最小 diff 思路（伪代码示意）

```python
# src/models/trap_wrappers.py
class LogitScaleWrapper(nn.Module):
    def __init__(self, backbone, scale=50.0):
        super().__init__()
        self.backbone = backbone
        self.scale = float(scale)
    def forward(self, x):
        return self.backbone(x) * self.scale

class InputRandomizationWrapper(nn.Module):
    def __init__(self, backbone, kind="pad_resize", pad=4, out_size=32):
        super().__init__()
        ...
    def forward(self, x):
        # 每次 forward 重采样 padding 位置 + resize；保持 [0,1]
        ...
```

```python
# src/attacks/eot.py
def eot_grad(model, x, y, loss_fn, eot_samples):
    grads = []
    for _ in range(eot_samples):
        x_ = x.detach().requires_grad_(True)
        loss = loss_fn(model(x_), y)
        grads.append(torch.autograd.grad(loss, x_, only_inputs=True)[0])
    return torch.stack(grads, dim=0).mean(dim=0)
```

```python
# src/models/factory.py
def build_model(..., wrappers=None):
    backbone = ...
    if normalize:
        m = NormalizeWrapper(backbone, mean, std)
    else:
        m = backbone
    for w in (wrappers or []):
        if w["kind"] == "logit_scale":
            m = LogitScaleWrapper(m, scale=w.get("scale", 50.0))
        elif w["kind"] == "input_random":
            m = InputRandomizationWrapper(m, **w)
    return m
```

### 验收断言（对应 plan-v2 §11）

- `Gap_over(Trap-A) > 0.20`（极小 smoke，2 epoch SmallCNN standard + LogitScaleWrapper(T=50)，PGD-20 robust 应明显 > APGD-DLR robust）。
- Trap-B 在「无 EOT 白盒」下 PGD robust 显著 > Square robust；EOT (>=10) 后 PGD robust 接近 Square robust。
- `tests/test_c1_traps.py` 全通过；既有 `tests/test_attacks_smoke.py`、`test_models.py` 全通过。

### 对现有 smoke 测试影响

- 现有 `tests/test_models.py` 仅检 4 个诚实模型 forward shape，不受影响。
- 现有 `tests/test_attacks_smoke.py` 验 fgsm/pgd shape & range，attack runner 接口扩参后必须**保持向后兼容**（`eot_samples` 默认 0）。
- `scripts/run_aalite.py` 新增 CLI 选项不能 break `tests/test_config.py` 的 `--help` 关键字检查（既有 flag 全保留）。

---

## C2 — best.pt 按 robust val 选

### 文件级改动

| 操作 | 路径 | 说明 |
|---|---|---|
| **修改** | `src/training/trainer.py` | 新增 `evaluate_robust(steps, subset_loader, eps)` → 返回 `{"pgd7_val_acc": ...}`；`train()` 内根据 `defense` + `best_criterion` 选 best：AT 类默认 `robust_val`，standard 保留 `clean_val`；分别保存 `best.pt` 与 `best_clean.pt` |
| **修改** | `src/training/trainer.py` | `TrainConfig` 新增 `best_criterion: str = "auto"`, `robust_val_steps: int = 7`, `robust_val_subset_size: int = 512`, `eval_every: int = 1` |
| **修改** | `src/training/trainer.py` | `metrics.json` 新增 `best_criterion, best_metric, best_metric_value, best_epoch, best_clean_acc, best_robust_acc` |
| **修改** | `configs/defenses/*_at.yaml` | 加 `best_criterion: robust_val`、`robust_val: {steps: 7, subset_size: 512}` |
| **修改** | `configs/defenses/standard.yaml` | 加 `best_criterion: clean_val`（保持旧行为） |
| **修改** | `scripts/train.py` | 加载 defense yaml 时把 `best_criterion / robust_val / eval_every` 透传到 `TrainConfig` |
| **新增** | `tests/test_c2_best_robust_val.py` | 极小 SmallCNN + fake data，2 epoch FGSM-AT → 断言 `metrics.json['best_criterion']=='robust_val'` 且 `'best_robust_acc' in metrics` |

### 最小 diff 思路

```python
# trainer.py
def train(self):
    ...
    best_robust = -1.0
    best_clean = -1.0
    best_epoch_robust = best_epoch_clean = 0
    use_robust_best = self.config.best_criterion == "robust_val"
    for epoch in range(self.config.epochs):
        train_stats = self.train_one_epoch(epoch)
        clean_stats = self.evaluate_clean()
        robust_stats = self.evaluate_robust(...) if use_robust_best and (epoch+1)%eval_every==0 else {}
        ...
        # save best by clean (always saved as best_clean.pt)
        if clean_stats["val_acc"] > best_clean:
            best_clean = clean_stats["val_acc"]; best_epoch_clean = epoch+1
            save_checkpoint(self.run_dir / "best_clean.pt", ckpt)
        # save best by robust if applicable
        if use_robust_best and robust_stats and robust_stats["pgd7_val_acc"] > best_robust:
            best_robust = robust_stats["pgd7_val_acc"]; best_epoch_robust = epoch+1
            save_checkpoint(self.run_dir / "best.pt", ckpt)
        # standard or fallback: use clean as best
        if not use_robust_best and clean_stats["val_acc"] > best_clean_for_best:
            ...
            save_checkpoint(self.run_dir / "best.pt", ckpt)
    metrics["best_criterion"] = "robust_val" if use_robust_best else "clean_val"
    metrics["best_metric"] = "pgd7_val_acc" if use_robust_best else "val_acc"
    metrics["best_metric_value"] = best_robust if use_robust_best else best_clean
    metrics["best_epoch"] = best_epoch_robust if use_robust_best else best_epoch_clean
    metrics["best_clean_acc"] = best_clean
    metrics["best_robust_acc"] = best_robust if best_robust >= 0 else None
```

### 验收断言

- `metrics.best_criterion == "robust_val"`（AT 类）/ `"clean_val"`（standard）。
- 极小 smoke：FGSM-AT 5 epoch，`best_epoch_robust` 不晚于 `best_epoch_clean - 1`（即捕捉到 robust 峰）。
- `tests/test_c2_best_robust_val.py` pass；`tests/test_training_smoke.py` 仍 pass。

### 对现有 smoke 测试影响

- `tests/test_training_smoke.py:24-40` 当前断言 `result["clean_acc"] >= 0.0` 与 `best.pt` 存在；改动后 `result` 应改读 `best_clean_acc` 或保留 `clean_acc` 别名（向后兼容）。建议保留 `clean_acc` 字段（= `best_clean_acc`）以不破坏测试。

---

## C3 — train_attack ≠ eval_attack 解耦

### 文件级改动

| 操作 | 路径 | 说明 |
|---|---|---|
| **修改** | `src/training/trainer.py` | `TrainConfig` 加 `train_attack: dict | None = None`；`_attack_batch` 优先用 `self.config.train_attack`，回落到旧 `pgd_steps/pgd_alpha`；`metrics.json` 写 `train_attack_config` 段 |
| **修改** | `scripts/train.py` | 新增 `--defense-config` 选项加载 `configs/defenses/<name>.yaml`，把 `train_attack` 段透传给 `TrainConfig` |
| **修改** | `configs/defenses/pgd_at.yaml` | 已有 `train_attack` ✓ 保持；新增 `eval: {attacks: [pgd20, apgd_ce_50, apgd_dlr_50]}` 段 |
| **修改** | `configs/defenses/fgsm_at.yaml` | 加 `train_attack: {name: fgsm_rs, eps: 8/255, alpha: 10/255, random_start: true}`（与 C4 配合）+ `eval:` 段 |
| **修改** | `configs/defenses/fixed_mixed_at.yaml` | 每 stage 显式写 `train_attack`（如 stage1 fgsm-rs；stage2 pgd-7；stage3 pgd-7 + apgd-ce-5）+ `eval:` 段 |
| **新增** | `tests/test_c3_train_eval_split.py` | (a) `train_attack.steps < eval pgd20 steps`；(b) trainer 实际读 `train_attack` 段（mock TrainConfig 验证 `_attack_batch` 用的步数=7 不是 20） |

### 最小 diff 思路

```python
# trainer.py
@dataclass
class TrainConfig:
    ...
    train_attack: dict | None = None   # NEW

def _attack_batch(self, images, labels, attack_name):
    if attack_name == "clean":
        return images
    cfg = self.config.train_attack or {}
    if attack_name == "fgsm":
        return fgsm_attack(self.model, images, labels, eps=cfg.get("eps", self.config.eps))
    if attack_name == "fgsm_rs":
        return fgsm_rs_attack(self.model, images, labels,
                              eps=cfg.get("eps", self.config.eps),
                              alpha=cfg.get("alpha", 10/255),
                              random_start=cfg.get("random_start", True))
    if attack_name == "pgd":
        steps = cfg.get("steps", self.config.pgd_steps)
        return pgd_attack(self.model, images, labels,
                          eps=cfg.get("eps", self.config.eps),
                          alpha=cfg.get("alpha", self.config.pgd_alpha),
                          steps=steps, restarts=1, random_start=True)
    ...
```

```python
# scripts/train.py
if args.defense_config:
    defense_cfg = load_yaml(args.defense_config)["training"]
    train_attack = defense_cfg.get("train_attack")
    best_criterion = defense_cfg.get("best_criterion", "auto")
    ...
```

### 验收断言

- 加载 `pgd_at.yaml` 训练时，`metrics.train_attack_config.steps == 7`；评测时 `pgd20 steps == 20`。
- `tests/test_c3_train_eval_split.py` pass。
- 旧路径（不传 `--defense-config`）仍按 `TrainConfig.pgd_steps=7` 走，旧 smoke 测试不变。

### 对现有 smoke 测试影响

- `tests/test_training_smoke.py` 不传 `--defense-config`，走旧默认，pass。
- `tests/test_config.py:25-46` 的 `--help` flag 关键字检查不受影响。

---

## C4 — FGSM-AT 改 FGSM-RS + CO 检测

### 文件级改动

| 操作 | 路径 | 说明 |
|---|---|---|
| **修改** | `src/attacks/factory.py` | 新增 `fgsm_rs_attack(model, x, y, eps, alpha, random_start=True)`；注册到 `AttackFactory.create` 的 `fgsm_rs` 分支 |
| **修改** | `src/training/trainer.py` | FGSM-AT 路径默认 attack name 改为 `fgsm_rs`（通过 `_mix_for_epoch` 与 defense yaml 协同）；新增 `_co_check(epoch)` 在每 `co_check_every` 跑 PGD-7 val，单轮跌幅 > `co_threshold` 触发 `co_detected=True`、`co_epoch=epoch`、提前 `break` 训练循环并保留当前最优 checkpoint |
| **修改** | `src/training/trainer.py` | `TrainConfig` 加 `co_check_enabled: bool = False, co_check_every: int = 1, co_threshold: float = 0.15`；`metrics.json` 加 `co_detected: bool, co_epoch: int | None, co_pgd7_history: list[float]` |
| **修改** | `configs/defenses/fgsm_at.yaml` | `train_attack: {name: fgsm_rs, eps: 8/255, alpha: 10/255, random_start: true}` + `co_check: {enabled: true, every: 1, threshold: 0.15}` |
| **新增** | `tests/test_c4_fgsm_rs_co.py` | (a) `fgsm_rs_attack` 输出 [0,1] 且与朴素 fgsm 输出不同（random init 生效）；(b) 模拟 PGD-7 val 序列 [0.40, 0.42, 0.10] → 触发 CO 检测 |

### 最小 diff 思路

```python
# src/attacks/factory.py
def fgsm_rs_attack(model, images, labels, eps=8/255, alpha=10/255, random_start=True, set_eval=True):
    images = images.detach()
    with temporary_eval(model, enabled=set_eval):
        if random_start:
            delta = torch.empty_like(images).uniform_(-eps, eps)
            x0 = (images + delta).clamp(0, 1)
        else:
            x0 = images.clone()
        x0 = x0.detach().requires_grad_(True)
        loss = F.cross_entropy(model(x0), labels)
        grad = torch.autograd.grad(loss, x0, only_inputs=True)[0]
        adv = x0 + alpha * grad.sign()
        adv = images + (adv - images).clamp(-eps, eps)
        adv = adv.clamp(0, 1)
    return adv.detach()
```

```python
# trainer.py
def _co_check(self, epoch, history):
    if not self.config.co_check_enabled or (epoch+1) % self.config.co_check_every != 0:
        return False, None
    pgd7 = self._evaluate_pgd7_val()
    history.append(pgd7)
    if len(history) >= 2 and history[-2] - history[-1] > self.config.co_threshold:
        return True, epoch + 1
    return False, None
```

### 验收断言

- FGSM-RS run：`metrics.co_detected ∈ {true, false}`；若 true，`co_epoch` 不为 None；如 false，需要满足"PGD-val 不坍塌到 ≈0"（在 real run 时判，smoke 不强求）。
- `tests/test_c4_fgsm_rs_co.py` pass。
- 旧 smoke（不开 `co_check`）路径不变。

### 对现有 smoke 测试影响

- 现有 `test_attacks_smoke.py` 不会触发 fgsm_rs，无影响。
- `test_training_smoke.py` 不开 co_check，无影响。

---

## C5 — eval_subset_id

### 文件级改动

| 操作 | 路径 | 说明 |
|---|---|---|
| **新增** | `configs/eval/subsets.yaml` | 定义命名子集：`aalite_2k: {size: 2000, seed: 0}`、`square_1k: {size: 1000, seed: 0}`、`full_test: {size: 0, seed: 0}` |
| **修改** | `src/datasets/cifar10.py` | 新增 `build_named_eval_loader(name, cfg, root, ...) -> (loader, subset_id, indices)`；`subset_id = sha1(f"{dataset_name}|{name}|{size}|{seed}").hexdigest()[:12]` |
| **修改** | `src/attacks/runner.py` | metadata 透传 `eval_subset_id`；放进 result payload |
| **修改** | `src/attacks/aalite.py` | payload 加 `eval_subset_id`；`gap_over` 计算前断言所有子攻击 ID 一致，不一致 → `gap_over = None, gap_over_error = "subset_id_mismatch"` |
| **修改** | `scripts/evaluate_attack.py` / `scripts/run_aalite.py` | 新增 `--eval-subset <name>`（如 `aalite_2k`）；与 `--subset-size` 互斥（前者优先） |
| **修改** | `scripts/aggregate_results.py` | 聚合时按 `exp_id` 分组，校验 `pgd20/apgd_ce/apgd_dlr` 三攻击的 `eval_subset_id` 相同；不一致写 `gap_over_status="invalid_subset"` 并保留单攻击 robust acc |
| **新增** | `tests/test_c5_subset_id.py` | 同 `name+size+seed` 多次生成 ID 一致；不同 size 生成 ID 不同；mismatch 时 `run_aalite` 输出 `gap_over_error` |

### 验收断言

- `R_lite/Gap_over` JSON 必含 `eval_subset_id`；同攻击两次跑同子集 ID 一致。
- 聚合 CSV 出现 `eval_subset_id` 列；不同攻击 ID 不一致触发警告。
- `tests/test_c5_subset_id.py` pass。

### 对现有 smoke 测试影响

- 旧 `--subset-size` 路径保留：未传 `--eval-subset` 时回落，但 `eval_subset_id = sha1("legacy|<size>|<seed>")` 用于 trace（不强校验）。
- `tests/test_attacks_smoke.py` 不传该选项，无影响。

---

## C6 — APGD steps=50 + Square

### 文件级改动

| 操作 | 路径 | 说明 |
|---|---|---|
| **修改** | `configs/attacks/apgd_ce.yaml` | `steps: 50`（restarts 维持 1） |
| **修改** | `configs/attacks/apgd_dlr.yaml` | `steps: 50` |
| **修改** | `src/attacks/factory.py` `build_attack_config` defaults | `apgd_ce/apgd_dlr` 默认 `steps=50` |
| **修改** | `scripts/run_aalite.py` `--steps` 默认改 50 | 同步 |
| **新增** | `configs/attacks/square.yaml` | `n_queries: 2000, subset_size: 1000, eps: 8/255` |
| **新增** | `src/attacks/square.py` | 优先 `from torchattacks import Square`；缺失 → `AttackUnavailableError` 让 runner 标 skipped（不伪造） |
| **修改** | `src/attacks/factory.py` `AttackFactory.create` | 新增 `square` 分支 |
| **修改** | `scripts/evaluate_attack.py` `--attack choices` | 加入 `square` |
| **新增** | `tests/test_c6_apgd_steps_square.py` | (a) `build_attack_config('apgd_ce')` 默认 steps=50；(b) Square 缺包时 runner status=skipped 不抛 |

### 验收断言

- 所有 APGD CLI / config 默认 steps=50。
- Square 缺包时不阻塞主路径。
- `tests/test_c6_apgd_steps_square.py` pass。

### 对现有 smoke 测试影响

- 旧 `--steps 2` 路径仍存在（CLI 优先 > config 默认），smoke 测试不变。

---

## C7 — R_lite 白盒-only 声明

### 文件级改动

| 操作 | 路径 | 说明 |
|---|---|---|
| **修改** | `README.md` | 在 `AA-Lite Definition` 段加段："**R_lite 仅由 PGD-20/APGD-CE/APGD-DLR 三个白盒攻击的最差值构成；不含黑盒；黑盒以 Square 子集 + 「黑盒强于白盒」诊断单独评估。**" |
| **修改** | `src/attacks/aalite.py` | payload 加 `"r_lite_scope": "whitebox", "blackbox_handled_separately": true, "blackbox_notes": "Square subset + diagnostic, see configs/attacks/square.yaml"` |
| **修改** | `scripts/aggregate_results.py` | `main_robustness.csv` 加列 `r_lite_scope`（值固定 whitebox） |
| **新增** | （并入 `tests/test_c1_traps.py` 末尾） | 断言 aalite payload 含 `r_lite_scope=="whitebox"` |

### 验收断言

- 输出 JSON 与 README 都有显式 whitebox 声明。

### 对现有 smoke 测试影响

- 新增字段不影响既有断言。

---

## C8 — 等 GPU 小时三列 + wall-clock 预算

### 文件级改动

| 操作 | 路径 | 说明 |
|---|---|---|
| **修改** | `src/training/trainer.py` `TrainConfig` | 加 `max_wall_seconds: int = 0`（0 = 无截断）、`amp: bool` 已有 ✓ |
| **修改** | `src/training/trainer.py` `train()` | 每 batch 末检查 `time.perf_counter() - start > max_wall_seconds` → break epoch + outer loop |
| **修改** | `src/training/trainer.py` `metrics.json` | 必写 `gpu_name`（CPU 则 `"cpu"`）、`batch_size`、`amp`、`session_id`（每个进程一份 uuid 或日期戳）、`max_wall_seconds` |
| **修改** | `scripts/train.py` | 加 `--max-wall-seconds` 选项 |
| **修改** | `scripts/aggregate_results.py` `budget_comparison.csv` | 加列 `gpu_name, batch_size, amp, session_id, max_wall_seconds`；若同一比较内（按 `experiment_group` 分组）三列不全一致 → `equal_budget_invalid=true` |
| **新增** | `tests/test_c8_budget_columns.py` | mock 两条 metrics.json（一个 batch=128, 一个 batch=64），聚合时该 `experiment_group` 的 `equal_budget_invalid==true` |

### 验收断言

- 所有 metrics.json 含 `gpu_name/batch_size/amp/session_id`。
- `budget_comparison.csv` 含这五列。
- `--max-wall-seconds 30` 强制 30 秒退出且 last.pt 完好。

### 对现有 smoke 测试影响

- 新增字段不破坏 `test_training_smoke.py`。
- `--help` 新增 flag 不破坏 `test_config.py`（既有 flag 全保留）。

---

## C9 — smoke / real 物理隔离

### 文件级改动

| 操作 | 路径 | 说明 |
|---|---|---|
| **修改** | `src/datasets/cifar10.py` | 新增参数 `mode: Literal["real","smoke"]="real"`；如果 `mode=="real"` 且 `dataset_name=="fake_cifar10"` → `raise RuntimeError("real mode forbids fake_cifar10")` |
| **修改** | `scripts/train.py` / `evaluate_clean.py` / `evaluate_attack.py` / `run_aalite.py` / `run_autoattack_subset.py` / `run_diagnostics.py` | 新增 `--mode {real,smoke}`，默认 `real`；输出根从 `results/raw/...` → `results/{mode}/raw/...`；run_diagnostics 额外补漏的 `--dataset` |
| **修改** | 所有 metrics.json / 攻击 JSON / aalite payload | 强制写 `mode` 与 `dataset_name` 字段（来自数据模块） |
| **修改** | `scripts/aggregate_results.py` | 默认 `--input` 改为 `results/real/raw`；遇 `mode=="smoke"` 或 `dataset_name=="fake_cifar10"` → `raise RuntimeError` 并打印冲突文件路径；新增 `--allow-smoke` 旁路开关（仅给 P6/dev 用） |
| **新增** | `configs/experiments/*_smoke.yaml`（5 文件）+ `*_real.yaml`（5 文件） | 把现有 5 个 g* yaml 复制并按模式裁配置（smoke：fake/tiny；real：cifar10/full） |
| **修改** | `README.md` | 顶部加 banner 段："P6 产物为 smoke，结果不入论文；real 实验请用 `--mode real` + `configs/experiments/g*_real.yaml`。" |
| **新增** | `tests/test_c9_mode_isolation.py` | (a) `build_dataloaders(mode='real', dataset_name='fake_cifar10')` 抛 RuntimeError；(b) aggregate 默认输入 `results/real/raw`；(c) 把一条 smoke JSON 放进 real 目录 → aggregate 抛错 |

### 验收断言

- `real` 模式下 fake 数据被禁止；`smoke` 与 `real` 输出根分离；real 聚合默认拒绝 smoke 标记。
- `tests/test_c9_mode_isolation.py` pass。
- 既有 smoke 测试可通过 `mode=smoke` 显式开启走旧逻辑。

### 对现有 smoke 测试影响

- `tests/test_training_smoke.py:11-40` 直接用 `CIFAR10DataModule(...,dataset_name="fake_cifar10")` 构造——需要在 datamodule 默认 `mode="smoke"`（否则会被 C9 断言拦死），或者在 test 里显式传 `mode="smoke"`。**推荐：DataModule 默认 `mode="real"`；调用方传 fake 时必须配 `mode="smoke"`；现有 test 显式补 `mode="smoke"`**——这是最安全的，把意图固化在 test 里。
- 现有 P6 写出的 fake JSON 已经在 `adv-robust-lite/results/raw/`，C9 实施时需要把它们物理迁移到 `adv-robust-lite/results/smoke/raw/`（提供 `scripts/migrate_results.py` 一次性脚本，或直接 git mv）。

---

## sanity gate 总览（plan-v2 §11 映射）

| sanity gate（plan-v2 §11） | 由哪个 C 实现 | 自动验收测试 |
|---|---|---|
| `best_criterion == "robust_val"` | C2 | `tests/test_c2_best_robust_val.py` |
| `train_attack.steps < eval pgd steps` | C3 | `tests/test_c3_train_eval_split.py` |
| `PGD-only.eval_subset_id == R_lite.eval_subset_id` | C5 | `tests/test_c5_subset_id.py` |
| `Gap_over(Trap-A) > Gap_over(诚实 AT)` | C1 | 极小 smoke：standard + LogitScaleWrapper 与 PGD-AT 对比 |
| Trap-B 触发「黑盒强于白盒」；诚实 AT 全通过 | C1 + C6 | smoke + diagnostics 联跑 |
| `eot_samples ≥ 10`（Trap-B 白盒评测） | C1 | `tests/test_c1_traps.py` |
| 诚实 PGD-AT 不崩 | 由 real run 验证，不在补丁 | — |
| smoke 不入 real | C9 | `tests/test_c9_mode_isolation.py` |

---

## 实施顺序与停-验-提交节奏

1. **C1**（最大改动；建议先在 `worktree` 上做） → smoke + 新 test → commit。
2. **C2** → smoke + 新 test → commit。
3. **C3**（依赖 C2 的 TrainConfig 字段扩展） → smoke + 新 test → commit。
4. **C4**（依赖 C3 的 train_attack 路径） → smoke + 新 test → commit。
5. **C5** → smoke + 新 test → commit。
6. **C6**（C1 Trap-B 需要 Square；建议 C1 后立刻做 C6） → 新 test → commit。
7. **C7** → README + payload 字段 → commit。
8. **C8** → 字段 + budget csv → 新 test → commit。
9. **C9**（最后做，最贴近发布；改 dataset module 易牵动现有 test） → 新 test → commit。
10. 收尾：`docs/ready_for_real_run.md`（列出从 smoke 到 real 需要改的 config 项与预计算力）。

---

## 失败回退

- 若 C1 的 EOT 性能太慢导致 smoke 也跑不动：先实现 `eot_samples<=4` 的轻量路径用于功能 smoke；real run 时再用 `>=10`。
- 若 C2 的 robust-val 在大模型上每 epoch 增量过大：把 `eval_every` 默认设到 5，并允许 standard 默认走 clean-val。
- 若 C9 的迁移导致旧 fake JSON 散落：提供 `scripts/migrate_results.py` 一次性挪到 `results/smoke/raw/`，并在 git commit message 标注。
