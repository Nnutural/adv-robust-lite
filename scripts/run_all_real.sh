#!/usr/bin/env bash
set -u
set -o pipefail

readonly EPOCHS_STANDARD="${EPOCHS_STANDARD:-50}"
readonly EPOCHS_AT="${EPOCHS_AT:-40}"
readonly PGD20_SUBSET="${PGD20_SUBSET:-5000}"
readonly AALITE_SUBSET="${AALITE_SUBSET:-1000}"
readonly AUTOATTACK_SUBSET="${AUTOATTACK_SUBSET:-500}"
readonly DIAG_SUBSET="${DIAG_SUBSET:-500}"
readonly SQUARE_SUBSET="${SQUARE_SUBSET:-500}"
readonly SQUARE_QUERIES="${SQUARE_QUERIES:-1000}"
readonly EOT_SAMPLES="${EOT_SAMPLES:-10}"
readonly G5_WALLCLOCK="${G5_WALLCLOCK:-1800}"
readonly SEED="${SEED:-0}"
readonly DEVICE="${DEVICE:-cuda}"
readonly BATCH="${BATCH:-128}"
readonly WORKERS="${WORKERS:-4}"

STAGES=(
  g1_mobilenet_train
  g1_mobilenet_eval
  g2_train_standard
  g2_train_fgsm_at
  g2_train_pgd_at
  g2_train_fixed_mixed_at
  g2_eval
  g3_aa_full_subset
  g3_trap_logit
  g3_trap_random_eot
  g3_trap_random_square
  g5_train_fgsm_at
  g5_train_pgd_at
  g5_train_fixed_mixed_at
  g5_eval
  p8_diagnostics
  aggregate
  figures
)

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/real_run_${RUN_ID}"
TIMING_FILE="${LOG_DIR}/stage_timing.tsv"
FAILURES_FILE="${LOG_DIR}/failures.txt"
mkdir -p "${LOG_DIR}"
printf "stage\twall_seconds\tstatus\n" > "${TIMING_FILE}"
: > "${FAILURES_FILE}"

START_STAGE="${START_STAGE:-g1_mobilenet}"
if [[ "${START_STAGE}" == "g1_mobilenet" ]]; then
  START_STAGE="g1_mobilenet_train"
fi
SKIP_STAGES="${SKIP_STAGES:-}"
RUN_STARTED=0
FAIL_COUNT=0

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

elapsed() {
  local start="$1"
  echo $(( $(date +%s) - start ))
}

is_skipped_stage() {
  local stage="$1"
  local item
  IFS=',' read -ra skip_items <<< "${SKIP_STAGES}"
  for item in "${skip_items[@]}"; do
    if [[ "${item}" == "${stage}" || "${stage}" == "${item}_"* ]]; then
      return 0
    fi
  done
  return 1
}

stage_log() {
  local index="$1"
  printf "%s/stage_%02d.log" "${LOG_DIR}" "${index}"
}

record_stage() {
  local stage="$1"
  local start="$2"
  local status="$3"
  printf "%s\t%s\t%s\n" "${stage}" "$(elapsed "${start}")" "${status}" >> "${TIMING_FILE}"
  if [[ "${status}" == "FAIL" ]]; then
    printf "%s\n" "${stage}" >> "${FAILURES_FILE}"
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
}

run_command() {
  local stage="$1"
  local logfile="$2"
  local rc
  shift 2
  printf "\n[%s] [%s] CMD:" "$(timestamp)" "${stage}" | tee -a "${logfile}"
  printf " %q" "$@" | tee -a "${logfile}"
  printf "\n" | tee -a "${logfile}"
  "$@" 2>&1 | tee -a "${logfile}"
  rc="${PIPESTATUS[0]}"
  printf "[%s] [%s] EXIT: %s\n" "$(timestamp)" "${stage}" "${rc}" | tee -a "${logfile}"
  return "${rc}"
}

require_ckpt() {
  local stage="$1"
  local logfile="$2"
  local ckpt="$3"
  if [[ ! -f "${ckpt}" ]]; then
    printf "[%s] [%s] [SKIP missing ckpt] %s\n" "$(timestamp)" "${stage}" "${ckpt}" | tee -a "${logfile}"
    return 1
  fi
  return 0
}

eval_common() {
  local stage="$1"
  local logfile="$2"
  local model="$3"
  local ckpt="$4"
  require_ckpt "${stage}" "${logfile}" "${ckpt}" || return 2
  local rc=0
  run_command "${stage}" "${logfile}" python scripts/evaluate_clean.py --mode real --dataset cifar10 --model "${model}" --checkpoint "${ckpt}" --batch-size "${BATCH}" --num-workers "${WORKERS}" --device "${DEVICE}" || rc=1
  run_command "${stage}" "${logfile}" python scripts/evaluate_attack.py --mode real --dataset cifar10 --model "${model}" --checkpoint "${ckpt}" --attack fgsm --batch-size "${BATCH}" --num-workers "${WORKERS}" --device "${DEVICE}" || rc=1
  run_command "${stage}" "${logfile}" python scripts/evaluate_attack.py --mode real --dataset cifar10 --model "${model}" --checkpoint "${ckpt}" --attack pgd20 --steps 20 --subset-size "${PGD20_SUBSET}" --batch-size "${BATCH}" --num-workers "${WORKERS}" --device "${DEVICE}" || rc=1
  run_command "${stage}" "${logfile}" python scripts/run_aalite.py --mode real --dataset cifar10 --model "${model}" --checkpoint "${ckpt}" --subset-size "${AALITE_SUBSET}" --batch-size "${BATCH}" --num-workers "${WORKERS}" --device "${DEVICE}" || rc=1
  return "${rc}"
}

print_precheck() {
  cat <<EOF
Log dir: ${LOG_DIR}
START_STAGE=${START_STAGE}
SKIP_STAGES=${SKIP_STAGES:-<none>}

Stages:
EOF
  local stage
  for stage in "${STAGES[@]}"; do
    printf "  - %s\n" "${stage}"
  done
  cat <<'EOF'

Checkpoint expectations:
  g1_mobilenet_eval: checkpoints/mobilenetv2_standard_seed0/best.pt
  g2_eval/p8: checkpoints/preact_resnet18_{standard,fgsm_at,pgd_at,fixed_mixed_at}_seed0/best.pt
  g3_aa_full_subset: checkpoints/preact_resnet18_{pgd_at,fixed_mixed_at}_seed0/best.pt
  g3_trap_*: checkpoints/preact_resnet18_standard_seed0/best.pt
  g5_eval: checkpoints/preact_resnet18_{fgsm_at,pgd_at,fixed_mixed_at}_seed1/best.pt
EOF
  printf "\nPress Enter to start, or Ctrl-C to abort..."
  read -r _
}

run_stage_body() {
  local stage="$1"
  local logfile="$2"
  local def ckpt rc missing
  case "${stage}" in
    g1_mobilenet_train)
      run_command "${stage}" "${logfile}" python scripts/train.py --mode real --dataset cifar10 --download --model mobilenetv2 --defense standard --epochs "${EPOCHS_STANDARD}" --batch-size "${BATCH}" --num-workers "${WORKERS}" --seed "${SEED}" --device "${DEVICE}" --amp --experiment-group g1_structure
      ;;
    g1_mobilenet_eval)
      eval_common "${stage}" "${logfile}" mobilenetv2 checkpoints/mobilenetv2_standard_seed0/best.pt
      ;;
    g2_train_standard)
      run_command "${stage}" "${logfile}" python scripts/train.py --mode real --dataset cifar10 --download --model preact_resnet18 --defense standard --epochs "${EPOCHS_STANDARD}" --batch-size "${BATCH}" --num-workers "${WORKERS}" --seed "${SEED}" --device "${DEVICE}" --amp --experiment-group g2_defense
      ;;
    g2_train_fgsm_at)
      run_command "${stage}" "${logfile}" python scripts/train.py --mode real --dataset cifar10 --download --model preact_resnet18 --defense fgsm_at --epochs "${EPOCHS_AT}" --batch-size "${BATCH}" --num-workers "${WORKERS}" --seed "${SEED}" --device "${DEVICE}" --amp --experiment-group g2_defense
      ;;
    g2_train_pgd_at)
      run_command "${stage}" "${logfile}" python scripts/train.py --mode real --dataset cifar10 --download --model preact_resnet18 --defense pgd_at --epochs "${EPOCHS_AT}" --batch-size "${BATCH}" --num-workers "${WORKERS}" --seed "${SEED}" --device "${DEVICE}" --amp --pgd-steps 7 --experiment-group g2_defense
      ;;
    g2_train_fixed_mixed_at)
      run_command "${stage}" "${logfile}" python scripts/train.py --mode real --dataset cifar10 --download --model preact_resnet18 --defense fixed_mixed_at --epochs "${EPOCHS_AT}" --batch-size "${BATCH}" --num-workers "${WORKERS}" --seed "${SEED}" --device "${DEVICE}" --amp --pgd-steps 7 --experiment-group g2_defense
      ;;
    g2_eval)
      rc=0
      missing=0
      for def in standard fgsm_at pgd_at fixed_mixed_at; do
        eval_common "${stage}" "${logfile}" preact_resnet18 "checkpoints/preact_resnet18_${def}_seed0/best.pt"
        case "$?" in
          0) ;;
          2) missing=1 ;;
          *) rc=1 ;;
        esac
      done
      [[ "${rc}" -eq 0 && "${missing}" -eq 1 ]] && return 2
      return "${rc}"
      ;;
    g3_aa_full_subset)
      rc=0
      missing=0
      for def in pgd_at fixed_mixed_at; do
        ckpt="checkpoints/preact_resnet18_${def}_seed0/best.pt"
        require_ckpt "${stage}" "${logfile}" "${ckpt}" || { missing=1; continue; }
        run_command "${stage}" "${logfile}" python scripts/run_autoattack_subset.py --mode real --dataset cifar10 --model preact_resnet18 --checkpoint "${ckpt}" --subset-size "${AUTOATTACK_SUBSET}" --batch-size "${BATCH}" --num-workers "${WORKERS}" --device "${DEVICE}" || rc=1
      done
      [[ "${rc}" -eq 0 && "${missing}" -eq 1 ]] && return 2
      return "${rc}"
      ;;
    g3_trap_logit)
      ckpt="checkpoints/preact_resnet18_standard_seed0/best.pt"
      require_ckpt "${stage}" "${logfile}" "${ckpt}" || return 2
      run_command "${stage}" "${logfile}" python scripts/run_aalite.py --mode real --dataset cifar10 --model preact_resnet18 --checkpoint "${ckpt}" --model-wrappers configs/defenses/trap_logit.yaml --subset-size "${AALITE_SUBSET}" --batch-size "${BATCH}" --num-workers "${WORKERS}" --device "${DEVICE}" --output results/real/raw/aalite/preact_resnet18_standard_seed0_trap_logit.json
      ;;
    g3_trap_random_eot)
      ckpt="checkpoints/preact_resnet18_standard_seed0/best.pt"
      require_ckpt "${stage}" "${logfile}" "${ckpt}" || return 2
      run_command "${stage}" "${logfile}" python scripts/run_aalite.py --mode real --dataset cifar10 --model preact_resnet18 --checkpoint "${ckpt}" --model-wrappers configs/defenses/trap_random.yaml --eot-samples "${EOT_SAMPLES}" --subset-size "${AALITE_SUBSET}" --batch-size "${BATCH}" --num-workers "${WORKERS}" --device "${DEVICE}" --output results/real/raw/aalite/preact_resnet18_standard_seed0_trap_random_eot.json
      ;;
    g3_trap_random_square)
      ckpt="checkpoints/preact_resnet18_standard_seed0/best.pt"
      require_ckpt "${stage}" "${logfile}" "${ckpt}" || return 2
      run_command "${stage}" "${logfile}" python scripts/evaluate_attack.py --mode real --dataset cifar10 --model preact_resnet18 --checkpoint "${ckpt}" --model-wrappers configs/defenses/trap_random.yaml --attack square --subset-size "${SQUARE_SUBSET}" --n-queries "${SQUARE_QUERIES}" --batch-size "${BATCH}" --num-workers "${WORKERS}" --device "${DEVICE}" --output results/real/raw/attacks/preact_resnet18_standard_seed0_trap_random_square.json
      ;;
    g5_train_fgsm_at)
      run_command "${stage}" "${logfile}" python scripts/train.py --mode real --dataset cifar10 --download --model preact_resnet18 --defense fgsm_at --epochs "${EPOCHS_AT}" --max-wall-seconds "${G5_WALLCLOCK}" --batch-size "${BATCH}" --num-workers "${WORKERS}" --seed 1 --device "${DEVICE}" --amp --experiment-group g5_equal_gpu_hours
      ;;
    g5_train_pgd_at)
      run_command "${stage}" "${logfile}" python scripts/train.py --mode real --dataset cifar10 --download --model preact_resnet18 --defense pgd_at --epochs "${EPOCHS_AT}" --max-wall-seconds "${G5_WALLCLOCK}" --batch-size "${BATCH}" --num-workers "${WORKERS}" --seed 1 --device "${DEVICE}" --amp --pgd-steps 7 --experiment-group g5_equal_gpu_hours
      ;;
    g5_train_fixed_mixed_at)
      run_command "${stage}" "${logfile}" python scripts/train.py --mode real --dataset cifar10 --download --model preact_resnet18 --defense fixed_mixed_at --epochs "${EPOCHS_AT}" --max-wall-seconds "${G5_WALLCLOCK}" --batch-size "${BATCH}" --num-workers "${WORKERS}" --seed 1 --device "${DEVICE}" --amp --pgd-steps 7 --experiment-group g5_equal_gpu_hours
      ;;
    g5_eval)
      rc=0
      missing=0
      for def in fgsm_at pgd_at fixed_mixed_at; do
        ckpt="checkpoints/preact_resnet18_${def}_seed1/best.pt"
        require_ckpt "${stage}" "${logfile}" "${ckpt}" || { missing=1; continue; }
        run_command "${stage}" "${logfile}" python scripts/run_aalite.py --mode real --dataset cifar10 --model preact_resnet18 --checkpoint "${ckpt}" --subset-size "${AALITE_SUBSET}" --batch-size "${BATCH}" --num-workers "${WORKERS}" --device "${DEVICE}" || rc=1
      done
      [[ "${rc}" -eq 0 && "${missing}" -eq 1 ]] && return 2
      return "${rc}"
      ;;
    p8_diagnostics)
      rc=0
      missing=0
      for def in standard fgsm_at pgd_at fixed_mixed_at; do
        ckpt="checkpoints/preact_resnet18_${def}_seed0/best.pt"
        require_ckpt "${stage}" "${logfile}" "${ckpt}" || { missing=1; continue; }
        run_command "${stage}" "${logfile}" python scripts/run_diagnostics.py --mode real --dataset cifar10 --model preact_resnet18 --checkpoint "${ckpt}" --eps-list 2/255,4/255,8/255,16/255 --steps-list 10,20,50 --restarts-list 1,3,5 --subset-size "${DIAG_SUBSET}" --batch-size "${BATCH}" --num-workers "${WORKERS}" --device "${DEVICE}" || rc=1
      done
      [[ "${rc}" -eq 0 && "${missing}" -eq 1 ]] && return 2
      return "${rc}"
      ;;
    aggregate)
      run_command "${stage}" "${logfile}" python scripts/aggregate_results.py --input results/real/raw --output results/real/tables
      ;;
    figures)
      run_command "${stage}" "${logfile}" python scripts/make_figures.py --tables results/real/tables --output results/real/figures
      ;;
    *)
      printf "[%s] Unknown stage: %s\n" "$(timestamp)" "${stage}" | tee -a "${logfile}"
      return 1
      ;;
  esac
}

run_stage() {
  local index="$1"
  local stage="$2"
  local logfile
  local start
  logfile="$(stage_log "${index}" "${stage}")"
  start="$(date +%s)"
  printf "\n[%s] ===== START %s =====\n" "$(timestamp)" "${stage}" | tee -a "${logfile}"
  if run_stage_body "${stage}" "${logfile}"; then
    printf "[%s] ===== OK %s =====\n" "$(timestamp)" "${stage}" | tee -a "${logfile}"
    record_stage "${stage}" "${start}" "OK"
  else
    local rc="$?"
    if [[ "${rc}" -eq 2 ]]; then
      printf "[%s] ===== SKIP %s =====\n" "$(timestamp)" "${stage}" | tee -a "${logfile}"
      record_stage "${stage}" "${start}" "SKIP"
    else
      printf "[%s] ===== FAIL %s =====\n" "$(timestamp)" "${stage}" | tee -a "${logfile}"
      record_stage "${stage}" "${start}" "FAIL"
    fi
  fi
}

print_precheck

index=0
for stage in "${STAGES[@]}"; do
  index=$((index + 1))
  if [[ "${RUN_STARTED}" -eq 0 ]]; then
    if [[ "${stage}" == "${START_STAGE}" ]]; then
      RUN_STARTED=1
    else
      continue
    fi
  fi
  if is_skipped_stage "${stage}"; then
    start="$(date +%s)"
    printf "[%s] [SKIP configured] %s\n" "$(timestamp)" "${stage}" | tee -a "$(stage_log "${index}" "${stage}")"
    record_stage "${stage}" "${start}" "SKIP"
    continue
  fi
  run_stage "${index}" "${stage}"
done

if [[ "${RUN_STARTED}" -eq 0 ]]; then
  printf "START_STAGE not found: %s\n" "${START_STAGE}" | tee -a "${FAILURES_FILE}"
  exit 1
fi

if [[ "${FAIL_COUNT}" -eq 0 ]]; then
  printf "\n全部完成 / 0 个 stage 失败\n"
else
  printf "\n全部完成 / %s 个 stage 失败，见 %s\n" "${FAIL_COUNT}" "${FAILURES_FILE}"
fi
