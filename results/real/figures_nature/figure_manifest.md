# Nature-style Figure Manifest

Backend: Python only (matplotlib/seaborn/pandas).
Archetype: quantitative grid with validation and appendix panels.
Export contract: PNG at 600 dpi plus editable PDF and SVG for every figure.
Data policy: existing real-result files only; no simulated data and no experiment reruns.

## figure_main_robustness
- Core conclusion: Standard training reaches the highest clean accuracy but has 0% PGD-20 and AA-Lite robustness, whereas adversarial training provides real robust accuracy.
- Source data: results/real/tables/main_robustness.csv
- Outputs: results/real/figures_nature/figure_main_robustness.png, results/real/figures_nature/figure_main_robustness.pdf, results/real/figures_nature/figure_main_robustness.svg

## figure_gap_over
- Core conclusion: For honest adversarial training, PGD-20 only slightly overestimates AA-Lite R_lite; trap rows, where present, are marked only as failed-trap limitations.
- Source data: results/real/tables/main_robustness.csv
- Outputs: results/real/figures_nature/figure_gap_over.png, results/real/figures_nature/figure_gap_over.pdf, results/real/figures_nature/figure_gap_over.svg

## figure_training_dynamics
- Core conclusion: FGSM-AT seed0 shows catastrophic robust-validation collapse at epoch 32, while PGD-AT and fixed mixed AT retain their best robust-validation checkpoints.
- Source data: checkpoints/*/train_log.csv and checkpoints/*/metrics.json
- Outputs: results/real/figures_nature/figure_training_dynamics.png, results/real/figures_nature/figure_training_dynamics.pdf, results/real/figures_nature/figure_training_dynamics.svg

## figure_cost_tradeoff
- Core conclusion: Positive GPU-hour adversarial-training runs improve AA-Lite robustness, with G2 and G5 points showing the cost and clean-accuracy trade-off.
- Source data: results/real/tables/budget_comparison.csv plus seed labels from results/real/tables/main_robustness.csv
- Outputs: results/real/figures_nature/figure_cost_tradeoff.png, results/real/figures_nature/figure_cost_tradeoff.pdf, results/real/figures_nature/figure_cost_tradeoff.svg

## figure_diagnostics
- Core conclusion: Real diagnostics pass epsilon, step, and restart sanity scans, supporting the main robustness measurements against gradient-masking concerns.
- Source data: results/real/raw/diagnostics/*.json
- Outputs: results/real/figures_nature/figure_diagnostics.png, results/real/figures_nature/figure_diagnostics.pdf, results/real/figures_nature/figure_diagnostics.svg

## figure_per_class_appendix
- Core conclusion: PGD-20 robustness varies strongly by CIFAR-10 class, with no adversarial-training recipe uniformly dominating every class.
- Source data: results/real/raw/attacks/*_pgd20_per_class.csv
- Outputs: results/real/figures_nature/figure_per_class_appendix.png, results/real/figures_nature/figure_per_class_appendix.pdf, results/real/figures_nature/figure_per_class_appendix.svg
