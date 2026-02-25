#!/usr/bin/env bash
set -euo pipefail

# This script runs patch-level EVA benchmarks (excluding cam16 and panda by default; uncomment below to enable).
# It is designed to be run on a single node with 1 GPU, run from within eva-probe subfolder of OpenMidnight.
# Outputs should match standard eva-probe, but this path is faster because embeddings stay in memory between
# `predict` and `fit` instead of being written to and reloaded from disk.
# Required positional args:
#   1) /absolute/path/to/teacher_checkpoint.pth
#   2) /absolute/path/to/output_root
# e.g., ./run_evals.sh /data/OpenMidnight_ckpts/openmidnight_checkpoint.pth /admin/home/paul/openmidnight_eval_results

DEFAULT_N_RUNS="${N_RUNS:-2}" # N_RUNS for everything except pcam (10shot) which is hardcoded to stay at 50 runs and cam16/panda which are hardcoded to 10 runs (down from default of 20); for final runs you should increase N_RUNS to 5 and modify 10->20 for cam16/panda
EVA_DATA_ROOT="/block/eva-data" # change to the directory that contains all the eva datasets 

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 /absolute/path/to/teacher_checkpoint.pth /absolute/path/to/output_root"
  exit 1
fi

CKPT_PATH="$1"
if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "Checkpoint not found: ${CKPT_PATH}"
  exit 1
fi

OUTPUT_ROOT="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
EVAL_CONFIG_DIR="${REPO_ROOT}/eval_configs"

source "${REPO_ROOT}/.venv/bin/activate"
cd "${SCRIPT_DIR}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_ROOT}/fast_eval_${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
MASTER_LOG="${RUN_DIR}/master.log"
STATUS_FILE="${RUN_DIR}/status.txt"
SUMMARY_TSV="${RUN_DIR}/summary.tsv"
RESULTS_ROOT="${RUN_DIR}/results"

mkdir -p "${LOG_DIR}"
mkdir -p "${RUN_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUM_DEVICES=1
export N_RUNS="${DEFAULT_N_RUNS}"
export N_DATA_WORKERS="${N_DATA_WORKERS:-8}"
export TQDM_REFRESH_RATE="${TQDM_REFRESH_RATE:-20}"
export CHECKPOINT_PATH="${CKPT_PATH}"
mkdir -p "${RESULTS_ROOT}"

printf "dataset\tmode\tstatus\telapsed_sec\tmetric\tmetric_mean\truns\tstart\tend\tresults_json\tlog_file\n" > "${SUMMARY_TSV}"

declare -a TASKS=(
  # classification patch-level tasks:
  "bach:predict_fit:${EVA_DATA_ROOT}/bach"
  "bracs:predict_fit:${EVA_DATA_ROOT}/bracs"
  "breakhist:predict_fit:${EVA_DATA_ROOT}/breakhis"
  "crc:predict_fit:${EVA_DATA_ROOT}/crc"
  "gleason_offline:predict_fit:${EVA_DATA_ROOT}/arvaniti_gleason_patches"
  "mhist:predict_fit:${EVA_DATA_ROOT}/mhist"
  "pcam_10:predict_fit:${EVA_DATA_ROOT}/patch_camelyon"
  "pcam:predict_fit:${EVA_DATA_ROOT}/patch_camelyon"
  # segmentation patch-level tasks:
  "consep:fit:${EVA_DATA_ROOT}/consep"
  "monusac:fit:${EVA_DATA_ROOT}/monusac"
  # whole slide tasks:
  "cam16_small:predict_fit:${EVA_DATA_ROOT}/camelyon16"
  "panda_small:predict_fit:${EVA_DATA_ROOT}/panda/prostate-cancer-grade-assessment"
)

total="${#TASKS[@]}"
done_count=0
ok_count=0
fail_count=0
failed_datasets=""

write_status() {
  local current_dataset="$1"
  local current_mode="$2"
  {
    echo "Run dir: ${RUN_DIR}"
    echo "Master log: ${MASTER_LOG}"
    echo "Summary TSV: ${SUMMARY_TSV}"
    echo "Progress: ${done_count}/${total} (ok=${ok_count}, fail=${fail_count})"
    echo "Current: ${current_dataset} (${current_mode})"
    echo "GPU: ${CUDA_VISIBLE_DEVICES}  N_RUNS=${N_RUNS}  NUM_DEVICES=${NUM_DEVICES}"
    echo ""
    cat "${SUMMARY_TSV}"
  } > "${STATUS_FILE}"
}

echo "Starting fast eval sweep at $(date -Iseconds)" | tee -a "${MASTER_LOG}"
echo "Using checkpoint: ${CKPT_PATH}" | tee -a "${MASTER_LOG}"
echo "Monitor with: watch -n 5 cat ${STATUS_FILE}" | tee -a "${MASTER_LOG}"
echo "Live log: tail -f ${MASTER_LOG}" | tee -a "${MASTER_LOG}"

write_status "initializing" "none"

for task in "${TASKS[@]}"; do
  IFS=':' read -r dataset mode data_root <<< "${task}"
  case "${dataset}" in
    cam16_small|panda_small) dataset_n_runs="10" ;;
    pcam_10) dataset_n_runs="50" ;;
    *) dataset_n_runs="${DEFAULT_N_RUNS}" ;;
  esac
  export N_RUNS="${dataset_n_runs}"
  config_path="${EVAL_CONFIG_DIR}/${dataset}.yaml"
  effective_mode="${mode}"
  effective_config_path="${config_path}"
  dataset_output_root="${RESULTS_ROOT}/${dataset}"
  cmd_string="N_RUNS=${dataset_n_runs} DATA_ROOT=${data_root} OUTPUT_ROOT=${dataset_output_root}"
  declare -a cmd_env=("N_RUNS=${dataset_n_runs}" "DATA_ROOT=${data_root}" "OUTPUT_ROOT=${dataset_output_root}")
  dataset_log="${LOG_DIR}/${dataset}.log"

  if [[ ! -f "${config_path}" ]]; then
    echo "Missing config: ${config_path}" | tee -a "${MASTER_LOG}"
    exit 1
  fi
  mkdir -p "${dataset_output_root}"

  start_epoch="$(date +%s)"
  start_iso="$(date -Iseconds)"
  write_status "${dataset}" "${effective_mode}"

  {
    echo ""
    echo "[$(date -Iseconds)] START ${dataset}"
    echo "mode=${effective_mode} data_root=${data_root}"
    echo "config=${effective_config_path}"
    echo "command=${cmd_string} eva ${effective_mode} --config ${effective_config_path}"
  } | tee -a "${MASTER_LOG}" "${dataset_log}"

  set +e
  env "${cmd_env[@]}" eva "${effective_mode}" --config "${effective_config_path}" 2>&1 | tee -a "${MASTER_LOG}" "${dataset_log}"
  rc="${PIPESTATUS[0]}"
  set -e

  end_epoch="$(date +%s)"
  end_iso="$(date -Iseconds)"
  elapsed="$((end_epoch - start_epoch))"
  done_count="$((done_count + 1))"
  metric_name="n/a"
  metric_mean="n/a"
  metric_runs="n/a"
  results_json_path="n/a"

  if [[ "${rc}" -eq 0 ]]; then
    status="ok"
    ok_count="$((ok_count + 1))"

    results_json_path="$(find "${dataset_output_root}" -type f -name 'results.json' | sort | tail -n 1)"
    if [[ -z "${results_json_path}" ]]; then
      echo "[$(date -Iseconds)] FAIL ${dataset} missing results.json under ${dataset_output_root}" | tee -a "${MASTER_LOG}" "${dataset_log}"
      status="fail"
      ok_count="$((ok_count - 1))"
      fail_count="$((fail_count + 1))"
      failed_datasets="${failed_datasets} ${dataset}"
    else
      case "${dataset}" in
        consep|monusac) metric_pattern="MonaiDiceScore" ;;
        bach|bracs|breakhist|crc|gleason_offline|panda_small) metric_pattern="MulticlassAccuracy" ;;
        *) metric_pattern="BinaryBalancedAccuracy" ;;
      esac
      mapfile -t metric_lines < <(python3 -c 'import json,sys; d=json.load(open(sys.argv[1], "r", encoding="utf-8")); pat=sys.argv[2]; matches=[(k,v["values"]) for stage in ("test","val") for ds in d["metrics"].get(stage,[]) for k,v in ds.items() if k.endswith(pat)]; metric,values=(matches[0] if matches else ("",[])); print(metric); print(", ".join(f"run_{i+1}={x:.6f}" for i,x in enumerate(values))); print(f"{sum(values)/len(values):.6f}" if values else "")' "${results_json_path}" "${metric_pattern}")
      if [[ "${#metric_lines[@]}" -lt 3 || -z "${metric_lines[0]}" || -z "${metric_lines[1]}" || -z "${metric_lines[2]}" ]]; then
        echo "[$(date -Iseconds)] FAIL ${dataset} metric ${metric_pattern} not found in ${results_json_path}" | tee -a "${MASTER_LOG}" "${dataset_log}"
        status="fail"
        ok_count="$((ok_count - 1))"
        fail_count="$((fail_count + 1))"
        failed_datasets="${failed_datasets} ${dataset}"
      else
        metric_name="${metric_lines[0]}"
        metric_mean="${metric_lines[2]}"
        metric_runs="${metric_lines[1]}"
      fi
    fi

    if [[ "${status}" == "ok" ]]; then
      echo "[$(date -Iseconds)] DONE ${dataset} elapsed=${elapsed}s" | tee -a "${MASTER_LOG}" "${dataset_log}"
    fi
  else
    status="fail"
    fail_count="$((fail_count + 1))"
    failed_datasets="${failed_datasets} ${dataset}"
    echo "[$(date -Iseconds)] FAIL ${dataset} rc=${rc} elapsed=${elapsed}s" | tee -a "${MASTER_LOG}" "${dataset_log}"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${dataset}" "${effective_mode}" "${status}" "${elapsed}" "${metric_name}" "${metric_mean}" "${metric_runs}" "${start_iso}" "${end_iso}" "${results_json_path}" "${dataset_log}" >> "${SUMMARY_TSV}"
  write_status "${dataset}" "${effective_mode}"
done

echo "" | tee -a "${MASTER_LOG}"
echo "Completed fast eval sweep at $(date -Iseconds)" | tee -a "${MASTER_LOG}"
echo "Summary: ok=${ok_count} fail=${fail_count}" | tee -a "${MASTER_LOG}"

if [[ "${fail_count}" -ne 0 ]]; then
  echo "Failed datasets:${failed_datasets}" | tee -a "${MASTER_LOG}"
  exit 1
fi

echo "All evals completed successfully." | tee -a "${MASTER_LOG}"
