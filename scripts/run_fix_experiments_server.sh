#!/usr/bin/env bash
set -u
set -o pipefail

readonly RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
readonly LOG_DIR="${LOG_DIR:-logs/run2_fix_${RUN_ID}}"
readonly FAILURES_FILE="${LOG_DIR}/failures.txt"
readonly COMMANDS_FILE="${LOG_DIR}/commands.tsv"

readonly DEVICE="${DEVICE:-cuda}"
readonly BATCH="${BATCH:-128}"
readonly WORKERS="${WORKERS:-4}"
readonly SEED="${SEED:-0}"
readonly PGD20_SUBSET="${PGD20_SUBSET:-5000}"
readonly AALITE_SUBSET="${AALITE_SUBSET:-1000}"
readonly AA_SUBSET="${AA_SUBSET:-500}"
readonly SQUARE_SUBSET="${SQUARE_SUBSET:-500}"
readonly SQUARE_QUERIES="${SQUARE_QUERIES:-1000}"
readonly EOT_SAMPLES="${EOT_SAMPLES:-10}"
readonly AA_SUBSET_INDICES="${AA_SUBSET_INDICES:-data/processed/aa_subset_indices_seed0.json}"

mkdir -p "${LOG_DIR}"
: > "${FAILURES_FILE}"
printf "index\tstage\tlabel\tlog_file\n" > "${COMMANDS_FILE}"

CMD_INDEX=0
FAIL_COUNT=0
NEXT_LOGFILE=""

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

safe_name() {
  local value="$1"
  value="${value//\//_}"
  value="${value// /_}"
  value="${value//:/_}"
  printf "%s" "${value}"
}

next_logfile() {
  local stage="$1"
  local label="$2"
  local safe_stage safe_label logfile
  CMD_INDEX=$((CMD_INDEX + 1))
  safe_stage="$(safe_name "${stage}")"
  safe_label="$(safe_name "${label}")"
  logfile="${LOG_DIR}/$(printf "%02d" "${CMD_INDEX}")_${safe_stage}_${safe_label}.log"
  printf "%02d\t%s\t%s\t%s\n" "${CMD_INDEX}" "${stage}" "${label}" "${logfile}" >> "${COMMANDS_FILE}"
  NEXT_LOGFILE="${logfile}"
}

record_failure() {
  local stage="$1"
  local label="$2"
  local rc="$3"
  local logfile="$4"
  FAIL_COUNT=$((FAIL_COUNT + 1))
  printf "%s\t%s\texit=%s\t%s\n" "${stage}" "${label}" "${rc}" "${logfile}" >> "${FAILURES_FILE}"
}

run_cmd() {
  local stage="$1"
  local label="$2"
  local logfile rc
  shift 2
  next_logfile "${stage}" "${label}"
  logfile="${NEXT_LOGFILE}"
  {
    printf "[%s] stage=%s label=%s\n" "$(timestamp)" "${stage}" "${label}"
    printf "[%s] pwd=%s\n" "$(timestamp)" "$(pwd)"
    printf "[%s] cmd:" "$(timestamp)"
    printf " %q" "$@"
    printf "\n"
  } | tee -a "${logfile}"
  "$@" 2>&1 | tee -a "${logfile}"
  rc="${PIPESTATUS[0]}"
  printf "[%s] exit=%s\n" "$(timestamp)" "${rc}" | tee -a "${logfile}"
  if [[ "${rc}" -ne 0 ]]; then
    record_failure "${stage}" "${label}" "${rc}" "${logfile}"
  fi
  return "${rc}"
}

run_shell_block() {
  local stage="$1"
  local label="$2"
  local logfile script rc
  next_logfile "${stage}" "${label}"
  logfile="${NEXT_LOGFILE}"
  script="$(cat)"
  {
    printf "[%s] stage=%s label=%s\n" "$(timestamp)" "${stage}" "${label}"
    printf "[%s] pwd=%s\n" "$(timestamp)" "$(pwd)"
    printf "[%s] shell block:\n%s\n" "$(timestamp)" "${script}"
  } | tee -a "${logfile}"
  bash -lc "${script}" 2>&1 | tee -a "${logfile}"
  rc="${PIPESTATUS[0]}"
  printf "[%s] exit=%s\n" "$(timestamp)" "${rc}" | tee -a "${logfile}"
  if [[ "${rc}" -ne 0 ]]; then
    record_failure "${stage}" "${label}" "${rc}" "${logfile}"
  fi
  return "${rc}"
}

print_overwrite_notice() {
  cat <<'EOF'
This script runs evaluation-only run-2 fix jobs. It does not train models.

Raw JSON outputs that may be overwritten:
  results/real/raw/clean/smallcnn_standard_seed0.json
  results/real/raw/clean/resnet18_standard_seed0.json
  results/real/raw/attacks/smallcnn_standard_seed0_fgsm.json
  results/real/raw/attacks/smallcnn_standard_seed0_pgd20.json
  results/real/raw/attacks/resnet18_standard_seed0_fgsm.json
  results/real/raw/attacks/resnet18_standard_seed0_pgd20.json
  results/real/raw/aalite/smallcnn_standard_seed0.json
  results/real/raw/aalite/resnet18_standard_seed0.json
  results/real/raw/aalite_aa_subset/preact_resnet18_pgd_at_seed0.json
  results/real/raw/aalite_aa_subset/preact_resnet18_fixed_mixed_at_seed0.json
  results/real/raw/aalite/preact_resnet18_standard_seed0_trap_logit.json
  results/real/raw/aalite/preact_resnet18_standard_seed0_trap_random_eot.json
  results/real/raw/attacks/preact_resnet18_standard_seed0_trap_random_square.json
  results/real/raw/attacks/preact_resnet18_standard_seed0_trap_random_square_per_class.csv

Derived outputs that may be overwritten:
  results/real/tables/main_robustness.csv
  results/real/tables/aa_subset_check.csv
  results/real/tables/gradient_masking_diagnostics.csv
  results/real/tables/budget_comparison.csv
  results/real/figures/*
  results/real/figures_nature/*
EOF
  printf "\nLogs will be written under: %s\n" "${LOG_DIR}"
  if [[ "${RUN2_FIX_ASSUME_YES:-0}" != "1" ]]; then
    printf "Type RUN2_FIX to continue: "
    read -r reply
    if [[ "${reply}" != "RUN2_FIX" ]]; then
      printf "Aborted by user confirmation gate.\n"
      exit 1
    fi
  fi
}

run_g1_model() {
  local model="$1"
  local ckpt="checkpoints/${model}_standard_seed0/best.pt"
  local stage="01_g1_${model}"
  run_cmd "${stage}" "clean" python scripts/evaluate_clean.py --mode real --dataset cifar10 --model "${model}" --checkpoint "${ckpt}" --batch-size "${BATCH}" --num-workers "${WORKERS}" --device "${DEVICE}"
  run_cmd "${stage}" "fgsm" python scripts/evaluate_attack.py --mode real --dataset cifar10 --model "${model}" --checkpoint "${ckpt}" --attack fgsm --batch-size "${BATCH}" --num-workers "${WORKERS}" --device "${DEVICE}"
  run_cmd "${stage}" "pgd20" python scripts/evaluate_attack.py --mode real --dataset cifar10 --model "${model}" --checkpoint "${ckpt}" --attack pgd20 --steps 20 --subset-size "${PGD20_SUBSET}" --batch-size "${BATCH}" --num-workers "${WORKERS}" --device "${DEVICE}"
  run_cmd "${stage}" "aalite" python scripts/run_aalite.py --mode real --dataset cifar10 --model "${model}" --checkpoint "${ckpt}" --subset-size "${AALITE_SUBSET}" --batch-size "${BATCH}" --num-workers "${WORKERS}" --device "${DEVICE}"
}

run_bias_same_subset() {
  local def ckpt output
  for def in pgd_at fixed_mixed_at; do
    ckpt="checkpoints/preact_resnet18_${def}_seed0/best.pt"
    output="results/real/raw/aalite_aa_subset/preact_resnet18_${def}_seed0.json"
    run_cmd "02_bias_aa_dev_aa" "${def}_aalite_aa_subset" python scripts/run_aalite.py --mode real --dataset cifar10 --model preact_resnet18 --checkpoint "${ckpt}" --subset-size "${AA_SUBSET}" --seed "${SEED}" --subset-indices-path "${AA_SUBSET_INDICES}" --batch-size "${BATCH}" --num-workers "${WORKERS}" --device "${DEVICE}" --output "${output}"
  done
}

run_trap() {
  local ckpt="checkpoints/preact_resnet18_standard_seed0/best.pt"
  run_cmd "03_trap" "trap_a_logit_aalite" python scripts/run_aalite.py --mode real --dataset cifar10 --model preact_resnet18 --checkpoint "${ckpt}" --model-wrappers configs/defenses/trap_logit.yaml --subset-size "${AALITE_SUBSET}" --batch-size "${BATCH}" --num-workers "${WORKERS}" --device "${DEVICE}" --output results/real/raw/aalite/preact_resnet18_standard_seed0_trap_logit.json
  run_cmd "03_trap" "trap_b_random_eot_aalite" python scripts/run_aalite.py --mode real --dataset cifar10 --model preact_resnet18 --checkpoint "${ckpt}" --model-wrappers configs/defenses/trap_random.yaml --eot-samples "${EOT_SAMPLES}" --subset-size "${AALITE_SUBSET}" --batch-size "${BATCH}" --num-workers "${WORKERS}" --device "${DEVICE}" --output results/real/raw/aalite/preact_resnet18_standard_seed0_trap_random_eot.json
  run_cmd "03_trap" "trap_b_square" python scripts/evaluate_attack.py --mode real --dataset cifar10 --model preact_resnet18 --checkpoint "${ckpt}" --model-wrappers configs/defenses/trap_random.yaml --attack square --subset-size "${SQUARE_SUBSET}" --n-queries "${SQUARE_QUERIES}" --batch-size "${BATCH}" --num-workers "${WORKERS}" --device "${DEVICE}" --output results/real/raw/attacks/preact_resnet18_standard_seed0_trap_random_square.json
}

run_final_check() {
  run_shell_block "07_final_check" "tables_figures_and_sanity" <<'EOF'
python - <<'PY'
import glob
import os
import pandas as pd

main = pd.read_csv("results/real/tables/main_robustness.csv")
aa = pd.read_csv("results/real/tables/aa_subset_check.csv")

print("\n[G1 rows]")
g1_ids = ["smallcnn_standard_seed0", "resnet18_standard_seed0", "mobilenetv2_standard_seed0"]
g1_cols = ["exp_id", "clean_acc", "fgsm_acc", "pgd20_acc", "pgd20_aalite_acc", "r_lite", "gap_over"]
print(main[main["exp_id"].isin(g1_ids)][g1_cols].to_string(index=False))
for exp_id in ["smallcnn_standard_seed0", "resnet18_standard_seed0"]:
    row = main.loc[main["exp_id"] == exp_id]
    assert not row.empty, f"missing G1 row: {exp_id}"
    for col in ["clean_acc", "fgsm_acc", "pgd20_acc", "pgd20_aalite_acc", "r_lite", "gap_over"]:
        assert row[col].notna().all(), f"{exp_id} has empty {col}"

print("\n[Trap rows]")
trap_cols = ["exp_id", "clean_acc", "pgd20_acc", "r_lite", "gap_over", "square_acc"]
trap = main[main["exp_id"].str.contains("trap", na=False)]
print(trap[trap_cols].to_string(index=False))
for exp_id in [
    "preact_resnet18_standard_seed0_trap_logit",
    "preact_resnet18_standard_seed0_trap_random_eot",
]:
    row = main.loc[main["exp_id"] == exp_id]
    assert not row.empty, f"missing Trap row: {exp_id}"
    clean = float(row["clean_acc"].iloc[0])
    assert clean > 0.5, f"{exp_id} clean_acc still looks like random guessing: {clean}"
random_row = main.loc[main["exp_id"] == "preact_resnet18_standard_seed0_trap_random_eot"]
assert random_row["square_acc"].notna().all(), "Trap-B square_acc missing"

print("\n[AA subset]")
aa_cols = ["exp_id", "r_lite_subset", "aa_subset_acc", "bias_aa", "dev_aa", "subset_size"]
print(aa[aa_cols].to_string(index=False))
for exp_id in ["preact_resnet18_pgd_at_seed0", "preact_resnet18_fixed_mixed_at_seed0"]:
    row = aa.loc[aa["exp_id"] == exp_id]
    assert not row.empty, f"missing AA subset row: {exp_id}"
    for col in ["r_lite_subset", "aa_subset_acc", "bias_aa", "dev_aa"]:
        assert row[col].notna().all(), f"{exp_id} has empty {col}"
    assert int(row["subset_size"].iloc[0]) == 500, f"{exp_id} subset_size is not 500"

print("\n[Standalone merge]")
for exp_id in [
    "preact_resnet18_fgsm_at_seed0",
    "preact_resnet18_pgd_at_seed0",
    "preact_resnet18_fixed_mixed_at_seed0",
]:
    row = main.loc[main["exp_id"] == exp_id]
    assert not row.empty, f"missing main row: {exp_id}"
    assert row["pgd20_source"].iloc[0] == "standalone_attack", f"{exp_id} pgd20_source mismatch"
    assert int(row["pgd20_subset_size"].iloc[0]) == 5000, f"{exp_id} PGD-20 subset mismatch"

print("\n[Figures]")
figure_files = [p for p in glob.glob("results/real/figures/*") + glob.glob("results/real/figures_nature/*") if os.path.isfile(p)]
for path in sorted(figure_files):
    size = os.path.getsize(path)
    print(path, size)
    assert size > 0, f"empty figure output: {path}"
assert any(path.endswith("figure_manifest.md") for path in figure_files), "missing Nature figure manifest"

print("\nFinal run-2 fix checks passed.")
PY
EOF
}

print_overwrite_notice

run_g1_model smallcnn
run_g1_model resnet18
run_bias_same_subset
run_trap
run_cmd "04_aggregate" "aggregate_results" python scripts/aggregate_results.py --input results/real/raw --output results/real/tables
run_cmd "05_figures" "make_figures" python scripts/make_figures.py --tables results/real/tables --output results/real/figures
run_cmd "06_nature_figures" "make_nature_figures" python scripts/make_nature_figures.py
run_final_check

if [[ "${FAIL_COUNT}" -eq 0 ]]; then
  printf "\nAll run-2 fix stages completed without recorded command failures.\n"
else
  printf "\nCompleted with %s recorded command failure(s). See %s\n" "${FAIL_COUNT}" "${FAILURES_FILE}"
fi
printf "Command index: %s\n" "${COMMANDS_FILE}"
