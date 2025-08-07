#!/bin/bash
usage() {
    echo "Usage: $0 -train_data <dir> -name <run_name> -model_dir <model_dir> [-lora]"
    exit 1
}

# -----------------------------------------------------------------------------
# Parse command‑line options
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -train_data|--train_data)   train_data="$2"; shift 2 ;;
        -name|--name)               run_name="$2";  shift 2 ;;
        -model_dir|--model_dir)     model_dir="$2"; shift 2 ;;
        -lora)                      lora=true;      shift   ;;
        -h|--help)                  usage ;;
        *)  echo "Unknown option: $1"; usage ;;
    esac
done

# -----------------------------------------------------------------------------
# Check required arguments
# -----------------------------------------------------------------------------
if [[ -z "$train_data" || -z "$run_name" ]]; then
    echo "Error: -train_data and -name are required."
    usage
fi

# -----------------------------------------------------------------------------
# Build and run the command
# -----------------------------------------------------------------------------
cmd="python phenogpt2_pretraining.py \
      -train_data \"${train_data}\" \
      -name \"${run_name}\""

[[ -n "$model_dir" ]] && cmd+=" -model_dir \"${model_dir}\""
[[ -n "$lora" ]] && cmd+=" -lora"

echo "Executing: $cmd"
eval "$cmd"

# -----------------------------------------------------------------------------
# Example sbatch submissions
# -----------------------------------------------------------------------------
# sbatch -p gpu-xe9680q --gres=gpu:h100:6 --cpus-per-gpu=3 --mem-per-cpu=50G \
#        --time=5-00:00:00 --profile=all --export=ALL --mail-type=ALL --mail-user=nguyenqm@chop.edu \
#        --wrap="bash pretrain_hpo_aware.sh \
#                  -train_data ./data/training_data/pretrain_data.pkl \
#                  -name hpo_pretrain_aware \
#                  -model_dir meta-llama/Llama-3.1-8B"
