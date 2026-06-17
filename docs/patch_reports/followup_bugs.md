# Follow-up Bugs

## 修改范围

- Bug 1：修改 `scripts/run_aalite.py`、`scripts/evaluate_attack.py`、`src/attacks/runner.py`、`src/attacks/aalite.py`，新增/透传 `eot_required` 元数据，并将 `eot_disabled_for_demo` 改为仅在 input-randomization 且 `eot_samples == 0` 时为 true。
- Bug 2：重写 `tests/test_c5_subset_id.py`，修复 `_DummyModel` 作用域问题，并真实覆盖 `gap_over_error == "subset_id_mismatch"` 分支。
- Bug 3：扩展 `tests/test_c1_traps.py`，新增不依赖 torchattacks 的 Trap-A PGD-CE numerical 断言。

## 新增/修改测试

- `test_eot_disabled_for_demo_is_false_for_non_randomized_models`
- `test_eot_disabled_for_demo_is_true_only_for_randomized_eot0`
- `test_trap_a_logit_scaling_inflates_pgd_ce_robust_acc`
- `test_aalite_gap_over_invalid_when_subset_ids_mismatch`
- `test_aalite_gap_over_ok_when_subset_ids_match`

## 验证

- `python -m pytest tests -q`：12 passed, 22 skipped。
- `python -m pytest tests/test_attacks_smoke.py -q`：3 skipped。
- `python -m pytest tests/test_training_smoke.py -q`、`tests/test_c2_best_robust_val.py -q`：本机无 torch，整文件 skipped，pytest 返回 1。
- 未改变任何攻击/训练数值产出：未改 PGD/FGSM/FGSM-RS/APGD、Trainer、optimizer、loss、step 数、checkpoint 格式或 metrics 字段名。
- 兼容性提示：旧 JSON 里 `eot_disabled_for_demo=true` 字段语义是错的；论文阶段如在意，可在新代码下重跑 AA-Lite 一次刷新该字段，不需要重训。
