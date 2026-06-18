# Run All Real Script

## 路径与用法

- 脚本：`scripts/run_all_real.sh`
- Linux 服务器实际运行：

```bash
cd /home/admin2/skq/Lesson/DLS/adv-robust-lite
conda activate skq_adv_exp
chmod +x scripts/run_all_real.sh
bash scripts/run_all_real.sh
```

- 从指定阶段继续：`START_STAGE=g2_eval bash scripts/run_all_real.sh`
- 跳过整组：`SKIP_STAGES=g5 bash scripts/run_all_real.sh`
- 精确跳过多段：`SKIP_STAGES=g5_train_fgsm_at,g5_train_pgd_at,g5_train_fixed_mixed_at bash scripts/run_all_real.sh`
- 能看到实时输出：每个 stage 会在终端打印时间戳、stage 名、命令、stdout/stderr 和退出码；内容同步 `tee` 到 `logs/real_run_*/stage_XX_<stage>.log`，行为类似 Makefile。
- 日志索引见 `logs/real_run_*/stage_map.tsv`；汇总见 `stage_timing.tsv` 和 `failures.txt`。
- 查看失败细节示例：`tail -n 120 logs/real_run_*/stage_03_g2_train_standard.log`。

## 参数与提醒

- 砍半参数：`EPOCHS_STANDARD` 50/80，`EPOCHS_AT` 40/80，`PGD20_SUBSET` 5000/10000，`AALITE_SUBSET` 1000/2000。
- 砍半参数：`AUTOATTACK_SUBSET` 500/1000，`DIAG_SUBSET` 500/1000，`SQUARE_SUBSET` 500/1000，`SQUARE_QUERIES` 1000/2000。
- 不砍参数：`EOT_SAMPLES` 10/10；预算参数：`G5_WALLCLOCK` 1800s。
- 估算总时间：RTX 3090 大约 6-8 h；RTX 3060 大约 12-15 h。
- Trap-B `EOT_SAMPLES=10` 不可再砍，这是 plan-v2 §11 sanity gate；`fixed_mixed_at` 会按相对进度自动缩到 40 ep，不需要改 yaml。
