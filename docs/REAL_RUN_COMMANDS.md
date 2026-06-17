# Real Run 命令清单

> **推荐**：一键调度脚本 `scripts/run_all_real.sh` 已按"课程级砍半参数"配置好，
> 直接 `bash scripts/run_all_real.sh` 即可从当前进度（已完成 G1 SmallCNN + ResNet-18 standard）
> 一路跑到论文出图。手动单条命令清单（本文档剩余部分）仅作参考。
>
> 砍半参数详见脚本顶部 readonly 变量；若要全口径运行，直接修改这些变量为 plan-v2 默认值。

本文档给出 G1-G5 与后续 P8 诊断/论文材料整理的可复制命令。所有命令都从仓库根目录执行：

```bash
cd /home/admin2/skq/Lesson/DLS/adv-robust-lite
```

Windows 本地路径对应：

```text
D:\Nnutural\Desktop\BUPT_6\深度学习与安全\期末大作业\adv-robust-lite
```

## 0. 环境检查

```bash
python -m pytest tests -q
python -m compileall -q src scripts tests
```

确认关键依赖：

```bash
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
try:
    import torchattacks
    print("torchattacks ok")
except Exception as e:
    print("torchattacks missing:", e)
try:
    import autoattack
    print("autoattack ok")
except Exception as e:
    print("autoattack missing:", e)
PY
```

如果 `autoattack` 不可从 PyPI 安装，可单独安装：

```bash
python -m pip install git+https://github.com/fra31/auto-attack.git
```

## 0.1 正式运行约定

- 所有正式论文实验都使用 `--mode real --dataset cifar10`。
- 正式结果只聚合 `results/real/raw`，输出到 `results/real/tables` 和 `results/real/figures`。
- 不要对正式聚合使用 `--allow-smoke`。
- `run_aalite.py` 不传 `--steps`：脚本默认会按 AA-Lite 设定运行 `PGD-20 + APGD-CE-50 + APGD-DLR-50`。如果传 `--steps 50`，会把 PGD-20 也改成 PGD-50，不符合本文实验定义。
- `torchattacks` 缺失时 APGD 会被记录为 skipped；正式实验应先安装并确认 `torchattacks ok`。

## 1. G1：结构鲁棒性实验

### 1.1 训练三个 standard 模型

```bash
python scripts/train.py --mode real --dataset cifar10 --download --model smallcnn --defense standard --epochs 80 --batch-size 128 --num-workers 4 --seed 0 --device cuda --amp --experiment-group g1_structure

python scripts/train.py --mode real --dataset cifar10 --download --model resnet18 --defense standard --epochs 80 --batch-size 128 --num-workers 4 --seed 0 --device cuda --amp --experiment-group g1_structure

python scripts/train.py --mode real --dataset cifar10 --download --model mobilenetv2 --defense standard --epochs 80 --batch-size 128 --num-workers 4 --seed 0 --device cuda --amp --experiment-group g1_structure
```

### 1.2 评测 SmallCNN

```bash
python scripts/evaluate_clean.py --mode real --dataset cifar10 --model smallcnn --checkpoint checkpoints/smallcnn_standard_seed0/best.pt --batch-size 128 --num-workers 4 --device cuda

python scripts/evaluate_attack.py --mode real --dataset cifar10 --model smallcnn --checkpoint checkpoints/smallcnn_standard_seed0/best.pt --attack fgsm --batch-size 128 --num-workers 4 --device cuda

python scripts/evaluate_attack.py --mode real --dataset cifar10 --model smallcnn --checkpoint checkpoints/smallcnn_standard_seed0/best.pt --attack pgd20 --steps 20 --subset-size 0 --batch-size 128 --num-workers 4 --device cuda

python scripts/run_aalite.py --mode real --dataset cifar10 --model smallcnn --checkpoint checkpoints/smallcnn_standard_seed0/best.pt --eval-subset aalite_2k --batch-size 128 --num-workers 4 --device cuda
```

### 1.3 评测 ResNet-18

```bash
python scripts/evaluate_clean.py --mode real --dataset cifar10 --model resnet18 --checkpoint checkpoints/resnet18_standard_seed0/best.pt --batch-size 128 --num-workers 4 --device cuda

python scripts/evaluate_attack.py --mode real --dataset cifar10 --model resnet18 --checkpoint checkpoints/resnet18_standard_seed0/best.pt --attack fgsm --batch-size 128 --num-workers 4 --device cuda

python scripts/evaluate_attack.py --mode real --dataset cifar10 --model resnet18 --checkpoint checkpoints/resnet18_standard_seed0/best.pt --attack pgd20 --steps 20 --subset-size 0 --batch-size 128 --num-workers 4 --device cuda

python scripts/run_aalite.py --mode real --dataset cifar10 --model resnet18 --checkpoint checkpoints/resnet18_standard_seed0/best.pt --eval-subset aalite_2k --batch-size 128 --num-workers 4 --device cuda
```

### 1.4 评测 MobileNetV2

```bash
python scripts/evaluate_clean.py --mode real --dataset cifar10 --model mobilenetv2 --checkpoint checkpoints/mobilenetv2_standard_seed0/best.pt --batch-size 128 --num-workers 4 --device cuda

python scripts/evaluate_attack.py --mode real --dataset cifar10 --model mobilenetv2 --checkpoint checkpoints/mobilenetv2_standard_seed0/best.pt --attack fgsm --batch-size 128 --num-workers 4 --device cuda

python scripts/evaluate_attack.py --mode real --dataset cifar10 --model mobilenetv2 --checkpoint checkpoints/mobilenetv2_standard_seed0/best.pt --attack pgd20 --steps 20 --subset-size 0 --batch-size 128 --num-workers 4 --device cuda

python scripts/run_aalite.py --mode real --dataset cifar10 --model mobilenetv2 --checkpoint checkpoints/mobilenetv2_standard_seed0/best.pt --eval-subset aalite_2k --batch-size 128 --num-workers 4 --device cuda
```

## 2. G2：防御泛化实验

### 2.1 训练 PreActResNet-18 四种防御

```bash
python scripts/train.py --mode real --dataset cifar10 --download --model preact_resnet18 --defense standard --epochs 80 --batch-size 128 --num-workers 4 --seed 0 --device cuda --amp --experiment-group g2_defense

python scripts/train.py --mode real --dataset cifar10 --download --model preact_resnet18 --defense fgsm_at --epochs 80 --batch-size 128 --num-workers 4 --seed 0 --device cuda --amp --experiment-group g2_defense

python scripts/train.py --mode real --dataset cifar10 --download --model preact_resnet18 --defense pgd_at --epochs 80 --batch-size 128 --num-workers 4 --seed 0 --device cuda --amp --pgd-steps 7 --experiment-group g2_defense

python scripts/train.py --mode real --dataset cifar10 --download --model preact_resnet18 --defense fixed_mixed_at --epochs 80 --batch-size 128 --num-workers 4 --seed 0 --device cuda --amp --pgd-steps 7 --experiment-group g2_defense
```

### 2.2 评测四个防御 checkpoint

```bash
for DEF in standard fgsm_at pgd_at fixed_mixed_at; do
  CKPT="checkpoints/preact_resnet18_${DEF}_seed0/best.pt"

  python scripts/evaluate_clean.py --mode real --dataset cifar10 --model preact_resnet18 --checkpoint "$CKPT" --batch-size 128 --num-workers 4 --device cuda

  python scripts/evaluate_attack.py --mode real --dataset cifar10 --model preact_resnet18 --checkpoint "$CKPT" --attack fgsm --batch-size 128 --num-workers 4 --device cuda

  python scripts/evaluate_attack.py --mode real --dataset cifar10 --model preact_resnet18 --checkpoint "$CKPT" --attack pgd20 --steps 20 --subset-size 0 --batch-size 128 --num-workers 4 --device cuda

  python scripts/run_aalite.py --mode real --dataset cifar10 --model preact_resnet18 --checkpoint "$CKPT" --eval-subset aalite_2k --batch-size 128 --num-workers 4 --device cuda
done
```

## 3. G4：Fixed Mixed-AT 组件/成本对比

当前可运行版本先比较已有三种防御：`fgsm_at`、`pgd_at`、`fixed_mixed_at`。

```bash
for DEF in fgsm_at pgd_at fixed_mixed_at; do
  CKPT="checkpoints/preact_resnet18_${DEF}_seed0/best.pt"
  python scripts/run_aalite.py --mode real --dataset cifar10 --model preact_resnet18 --checkpoint "$CKPT" --eval-subset aalite_2k --batch-size 128 --num-workers 4 --device cuda
done
```

注意：P7 中提到的 `fixed_mixed_at_no_apgd` 当前不是 `scripts/train.py --defense` 的合法选项。若后续实现该 defense，再补充：

```bash
python scripts/train.py --mode real --dataset cifar10 --download --model preact_resnet18 --defense fixed_mixed_at_no_apgd --epochs 80 --batch-size 128 --num-workers 4 --seed 0 --device cuda --amp --pgd-steps 7 --experiment-group g4_mixed_ablation

python scripts/run_aalite.py --mode real --dataset cifar10 --model preact_resnet18 --checkpoint checkpoints/preact_resnet18_fixed_mixed_at_no_apgd_seed0/best.pt --eval-subset aalite_2k --batch-size 128 --num-workers 4 --device cuda
```

## 4. G3：AA-Lite 评测协议消融

### 4.1 对四个防御模型执行 AA-Lite

```bash
for DEF in standard fgsm_at pgd_at fixed_mixed_at; do
  CKPT="checkpoints/preact_resnet18_${DEF}_seed0/best.pt"
  python scripts/run_aalite.py --mode real --dataset cifar10 --model preact_resnet18 --checkpoint "$CKPT" --eval-subset aalite_2k --batch-size 128 --num-workers 4 --device cuda
done
```

### 4.2 可选完整 AutoAttack 子集

建议先只对 PGD-AT 和 Fixed Mixed-AT 执行：

```bash
python scripts/run_autoattack_subset.py --mode real --dataset cifar10 --model preact_resnet18 --checkpoint checkpoints/preact_resnet18_pgd_at_seed0/best.pt --subset-size 1000 --batch-size 128 --num-workers 4 --device cuda

python scripts/run_autoattack_subset.py --mode real --dataset cifar10 --model preact_resnet18 --checkpoint checkpoints/preact_resnet18_fixed_mixed_at_seed0/best.pt --subset-size 1000 --batch-size 128 --num-workers 4 --device cuda
```

如果太慢，改为：

```bash
--subset-size 500
```

### 4.3 可选 Trap 诊断

```bash
python scripts/run_aalite.py --mode real --dataset cifar10 --model preact_resnet18 --checkpoint checkpoints/preact_resnet18_standard_seed0/best.pt --model-wrappers configs/defenses/trap_logit.yaml --eval-subset aalite_2k --batch-size 128 --num-workers 4 --device cuda

python scripts/run_aalite.py --mode real --dataset cifar10 --model preact_resnet18 --checkpoint checkpoints/preact_resnet18_standard_seed0/best.pt --model-wrappers configs/defenses/trap_random.yaml --eval-subset aalite_2k --eot-samples 10 --batch-size 128 --num-workers 4 --device cuda

python scripts/evaluate_attack.py --mode real --dataset cifar10 --model preact_resnet18 --checkpoint checkpoints/preact_resnet18_standard_seed0/best.pt --model-wrappers configs/defenses/trap_random.yaml --attack square --subset-size 1000 --n-queries 2000 --batch-size 128 --num-workers 4 --device cuda
```

## 5. G5：等 epoch / 等 GPU-hour 成本对比

### 5.1 等 epoch 聚合

如果只做等 epoch，直接聚合已有训练 metrics 和 AA-Lite：

```bash
python scripts/aggregate_results.py --input results/real/raw --output results/real/tables
python scripts/make_figures.py --tables results/real/tables --output results/real/figures
```

### 5.2 等 GPU-hour 训练

如果要做等 GPU-hour，需要统一 `--max-wall-seconds`、GPU、batch size、AMP 和 `--experiment-group`。示例：

```bash
python scripts/train.py --mode real --dataset cifar10 --download --model preact_resnet18 --defense fgsm_at --epochs 80 --max-wall-seconds 14400 --batch-size 128 --num-workers 4 --seed 1 --device cuda --amp --experiment-group g5_equal_gpu_hours

python scripts/train.py --mode real --dataset cifar10 --download --model preact_resnet18 --defense pgd_at --epochs 80 --max-wall-seconds 14400 --batch-size 128 --num-workers 4 --seed 1 --device cuda --amp --pgd-steps 7 --experiment-group g5_equal_gpu_hours

python scripts/train.py --mode real --dataset cifar10 --download --model preact_resnet18 --defense fixed_mixed_at --epochs 80 --max-wall-seconds 14400 --batch-size 128 --num-workers 4 --seed 1 --device cuda --amp --pgd-steps 7 --experiment-group g5_equal_gpu_hours
```

然后评测：

```bash
for DEF in fgsm_at pgd_at fixed_mixed_at; do
  CKPT="checkpoints/preact_resnet18_${DEF}_seed1/best.pt"
  python scripts/run_aalite.py --mode real --dataset cifar10 --model preact_resnet18 --checkpoint "$CKPT" --eval-subset aalite_2k --batch-size 128 --num-workers 4 --device cuda
done
```

最后聚合：

```bash
python scripts/aggregate_results.py --input results/real/raw --output results/real/tables
python scripts/make_figures.py --tables results/real/tables --output results/real/figures
```

## 6. P8：诊断与论文材料包

### 6.1 梯度遮蔽诊断

对四个防御模型运行诊断：

```bash
for DEF in standard fgsm_at pgd_at fixed_mixed_at; do
  CKPT="checkpoints/preact_resnet18_${DEF}_seed0/best.pt"

  python scripts/run_diagnostics.py --mode real --dataset cifar10 --model preact_resnet18 --checkpoint "$CKPT" --diagnostics epsilon --eps-list 2/255,4/255,8/255,16/255 --subset-size 1000 --batch-size 128 --num-workers 4 --device cuda

  python scripts/run_diagnostics.py --mode real --dataset cifar10 --model preact_resnet18 --checkpoint "$CKPT" --diagnostics steps --steps-list 10,20,50 --subset-size 1000 --batch-size 128 --num-workers 4 --device cuda

  python scripts/run_diagnostics.py --mode real --dataset cifar10 --model preact_resnet18 --checkpoint "$CKPT" --diagnostics restarts --restarts-list 1,3,5 --subset-size 1000 --batch-size 128 --num-workers 4 --device cuda
done
```

### 6.2 完整 AutoAttack 子集校验

```bash
python scripts/run_autoattack_subset.py --mode real --dataset cifar10 --model preact_resnet18 --checkpoint checkpoints/preact_resnet18_pgd_at_seed0/best.pt --subset-size 1000 --batch-size 128 --num-workers 4 --device cuda

python scripts/run_autoattack_subset.py --mode real --dataset cifar10 --model preact_resnet18 --checkpoint checkpoints/preact_resnet18_fixed_mixed_at_seed0/best.pt --subset-size 1000 --batch-size 128 --num-workers 4 --device cuda
```

### 6.3 最终聚合和出图

```bash
python scripts/aggregate_results.py --input results/real/raw --output results/real/tables
python scripts/make_figures.py --tables results/real/tables --output results/real/figures
```

检查输出：

```bash
ls -R results/real/tables
ls -R results/real/figures
```

## 7. 最低保底执行顺序

如果算力或时间有限，按下面顺序保留：

```text
G1 smallcnn/resnet18/mobilenetv2 standard
→ G2 preact_resnet18 standard/fgsm_at/pgd_at/fixed_mixed_at
→ 每个 checkpoint 跑 run_aalite
→ aggregate_results
→ make_figures
→ diagnostics 只对 preact_resnet18 fixed_mixed_at 跑
→ AutoAttack subset 只对 pgd_at 和 fixed_mixed_at 跑，必要时 subset-size=500
```

最终一定使用 real 路径聚合：

```bash
python scripts/aggregate_results.py --input results/real/raw --output results/real/tables
python scripts/make_figures.py --tables results/real/tables --output results/real/figures
```

不要使用 `--allow-smoke` 生成论文表格。
