# run-2 fix: aggregate, Trap wrapper, Bias_AA/Dev_AA

1. Modified files: `scripts/aggregate_results.py`, `scripts/run_aalite.py`, `scripts/evaluate_attack.py`, `src/models/factory.py`.
2. Aggregate now merges `results/real/raw/attacks/*_{fgsm,pgd20,square}.json` by experiment id; `fgsm_acc` uses standalone FGSM and `pgd20_acc` uses standalone PGD-20 when available.
3. `pgd20_aalite_acc`, `pgd20_source`, `pgd20_subset_size`, `fgsm_subset_size`, `square_acc`, and `square_subset_size` are written; failed/skipped standalone attacks are ignored.
4. `gap_over` is recomputed from `pgd20_aalite_acc - r_lite`, so FGSM-AT/PGD-AT/fixed_mixed_at remain AA-Lite same-subset values: 0.009/0.031/0.001.
5. Trap loading is fixed by loading checkpoints into an unwrapped base model first, then applying wrappers with `apply_model_wrappers()`.
6. `build_model()` reuses `apply_model_wrappers()`; attack and AA-Lite scripts now use the load-then-wrap order without changing wrapper or attack behavior.
7. Local smoke wrapper validation passed with `missing=0`, `unexpected=0`, and wrapped forward shape `(2, 10)`.
8. Exact Trap real rerun was not executed locally because `checkpoints/preact_resnet18_standard_seed0/best.pt` is absent here.
9. Existing Trap JSON still shows invalid old clean values (`trap_logit=0.104`, `trap_random_eot=0.096`); do not cite these as science results.
10. `run_aalite.py` and `evaluate_attack.py` now accept `--subset-indices-path` for fixed-index evaluation subsets.
11. `aggregate_results.py` reads `results/real/raw/aalite_aa_subset/*.json` only for `aa_subset_check.csv`, never for `main_robustness.csv`.
12. Synthetic same-subset aggregation validation passed: `r_lite_subset=0.36`, `aa_subset_acc=0.34`, `bias_aa=0.02`, `dev_aa=0.02`.
13. Current real `aa_subset_check.csv` still has empty Bias_AA/Dev_AA because no real `aalite_aa_subset/*.json` exists locally.
14. PGD-AT current row: `r_lite_subset=NaN`, `aa_subset_acc=0.3360000253`, `bias_aa=NaN`, `dev_aa=NaN`.
15. fixed_mixed_at current row: `r_lite_subset=NaN`, `aa_subset_acc=0.3400000036`, `bias_aa=NaN`, `dev_aa=NaN`.
16. After server same-subset reruns to `results/real/raw/aalite_aa_subset/{exp_id}.json`, both rows should fill automatically.
17. G1 local table currently only has `mobilenetv2_standard_seed0`; `smallcnn_standard_seed0` and `resnet18_standard_seed0` need server evaluation outputs.
18. Required G1 reruns: clean, FGSM, PGD-20 5k, and AA-Lite 1k for SmallCNN and ResNet-18.
19. Final local commands succeeded: `py_compile`, targeted `pytest`, aggregate, `make_figures.py`, and `make_nature_figures.py`.
20. Server rerun commands prepared for G1, Trap-A/B, Trap random Square, and AA-Lite AA-subset; run only evaluation, aggregation, and figure generation.
21. No checkpoint was retrained.
22. No Trainer, optimizer, loss, PGD/FGSM/APGD/Square numeric logic, or training/attack algorithm code was changed.
