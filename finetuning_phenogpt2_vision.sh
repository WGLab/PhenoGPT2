#!/bin/bash
#SBATCH --job-name=phenogpt2_vision
module load CUDA/12.1.1

usage() {
    echo "Usage: $0 -train_data <dir> -eval_data <dir> -name <run_name> -model_dir <model_dir> [-styles] [-lora]"
    exit 1
}

# -----------------------------------------------------------------------------
# Parse command‑line options
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -train_data|--train_data)   train_data="$2"; shift 2 ;;
        -eval_data|--eval_data)     eval_data="$2"; shift 2 ;;
        -name|--name)               run_name="$2";  shift 2 ;;
        -model_dir|--model_dir)               model_dir="$2";  shift 2 ;;
        -styles)                    styles=true;    shift   ;;
        -lora)                      lora=true;      shift   ;;
        -h|--help)                  usage ;;
        *)  echo "Unknown option: $1"; usage ;;
    esac
done

# -----------------------------------------------------------------------------
# Check required arguments
# -----------------------------------------------------------------------------
if [[ -z "$train_data" || -z "$eval_data" || -z "$run_name" ]]; then
    echo "Error: -train_data, -eval_data, and -name are required."
    usage
fi

# -----------------------------------------------------------------------------
# Build and run the command
# -----------------------------------------------------------------------------
cmd="python phenogpt2_vision_training.py \
      -train_data ${train_data} \
      -eval_data  ${eval_data} \
      -name       ${run_name}"

[[ -n "$styles" ]] && cmd+=" -styles"
[[ -n "$model_dir" ]] && cmd+=" -model_dir \"${model_dir}\""
[[ -n "$lora"   ]] && cmd+=" -lora"

echo "Executing: $cmd"
eval "$cmd"

# -----------------------------------------------------------------------------
# Example sbatch submissions
# -----------------------------------------------------------------------------
# sbatch -p gpuq --gres=gpu:a100:4 --cpus-per-gpu=3 --mem-per-cpu=50G \
#        --time=5-00:00:00 --profile=all --export=ALL --export=ALL --mail-type=ALL --mail-user=nguyenqm@chop.edu \
#        --wrap="bash finetuning_phenogpt2_vision.sh \
#                  -train_data ./data/vision_training/training_images.pkl \
#                  -eval_data ./data/vision_training/val_images.pkl \
#                  -name PhenoGPT2_Vision \
#                  -model_dir meta-llama/Llama-3.2-11B-Vision \
#                  -lora"