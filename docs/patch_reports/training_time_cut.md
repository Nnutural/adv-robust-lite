# Training Time Cut

- 新增变量：`MAX_TRAIN_BATCHES_AT=100`、`EPOCHS_PGD_AT=25`、`EPOCHS_FIXED_MIXED_AT=30`。
- 修改 5 个训练 stage：`g2_train_pgd_at`、`g2_train_fixed_mixed_at`、`g5_train_fgsm_at`、`g5_train_pgd_at`、`g5_train_fixed_mixed_at`。
- 这 5 个 stage 都加了 `--max-train-batches ${MAX_TRAIN_BATCHES_AT}`；PGD-AT 用 25 ep，fixed_mixed_at 用 30 ep。
- `g2_train_fgsm_at` 未改，保留当前正在跑的 40 ep 全训练集设置。
- FGSM-AT 跑完后建议用：`START_STAGE=g2_train_pgd_at bash scripts/run_all_real.sh` 接续。
