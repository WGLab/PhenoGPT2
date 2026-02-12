#!/bin/bash
set -euo pipefail

########################################
# Usage
########################################
usage() {
    echo "Usage: $0 \
--pretrain_model <model_dir> \
--train_data <train.pkl> \
--val_data <val.pkl> \
--output_dir <output_dir> \
--attn_implementation <attn_implementation>"
    exit 1
}

########################################
# Parse arguments
########################################
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --pretrain_model) pretrain_model="$2"; shift ;;
        --train_data)     train_data="$2"; shift ;;
        --val_data)       val_data="$2"; shift ;;
        --output_dir)     output_dir="$2"; shift ;;
        --attn_implementation)     attn_implementation="$2"; shift ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
    shift
done

########################################
# Validate required args
########################################
if [[ -z "${pretrain_model:-}" || -z "${train_data:-}" || -z "${output_dir:-}" ]]; then
    echo "ERROR: --pretrain_model, --train_data, and --output_dir are required"
    usage
fi

########################################
# GPU detection (Slurm-safe)
########################################
NUM_GPUS="${SLURM_GPUS_ON_NODE:-1}"
echo "Detected NUM_GPUS=${NUM_GPUS}"

########################################
# Deterministic, collision-proof port
########################################
BASE_PORT=12000

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    MASTER_PORT=$((BASE_PORT + SLURM_JOB_ID % 20000))
else
    MASTER_PORT=$((BASE_PORT + $$ % 20000))
fi

MASTER_ADDR="127.0.0.1"

echo "Using MASTER_ADDR=${MASTER_ADDR}"
echo "Using MASTER_PORT=${MASTER_PORT}"

########################################
# Triton cache (avoid NFS issues)
########################################
export TRITON_CACHE_DIR="/tmp/${USER}/triton_cache"
mkdir -p "${TRITON_CACHE_DIR}"

########################################
# NCCL sanity
########################################
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

########################################
# DeepSpeed config (EDIT IF NEEDED)
########################################
DS_CONFIG="./ds_z3.json"

########################################
# Logging
########################################
echo "========================================"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-N/A}"
echo "Start time: $(date)"
echo "Pretrain model: ${pretrain_model}"
echo "Train data: ${train_data}"
echo "Val data: ${val_data:-NONE}"
echo "Output dir: ${output_dir}"
echo "DeepSpeed config: ${DS_CONFIG}"
echo "attn_implementation: ${attn_implementation}"
echo "========================================"

########################################
# Build command (NO python here)
########################################
CMD="phenogpt2_training.py \
  --pretrain_model ${pretrain_model} \
  --train_data ${train_data} \
  --output_dir ${output_dir} \
  --attn_implementation ${attn_implementation} \
  --deepspeed ${DS_CONFIG}"

[[ -n "${val_data:-}" ]] && CMD="${CMD} --val_data ${val_data}"

########################################
# Launch DeepSpeed (FORCED PORT)
########################################
echo "========================================"
echo "Launching DeepSpeed"
echo "Command:"
echo "deepspeed --num_gpus=${NUM_GPUS} \
--master_addr ${MASTER_ADDR} \
--master_port ${MASTER_PORT} \
${CMD}"
echo "========================================"

deepspeed \
  --num_gpus="${NUM_GPUS}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  ${CMD}

echo "End time: $(date)"

# sbatch   -p gpu-xe9680q   --job-name=phenogpt2_qwen3_ft   --gres=gpu:h100:8   --cpus-per-gpu=2   --mem-per-cpu=50G   --time=5-00:00:00   --export=ALL   --mail-type=ALL   --mail-user=nguyenqm@chop.edu   finetuning_phenogpt2_text.sh   --pretrain_model HPO_AWARE_PRETRAIN_DIR   --train_data TRAIN_DATA_DIR   --val_data VAL_DATA_DIR    --output_dir OUTPUT_DIR --attn_implementation flash_attention_2