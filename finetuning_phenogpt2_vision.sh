#!/bin/bash
#SBATCH --job-name=phenogpt2_vision
set -euo pipefail

usage() {
    echo "Usage: $0 -train_data <dir> -eval_data <dir> -model <dir> -name <run_name> [-styles] [-lora]"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -train_data|--train_data)   train_data="$2"; shift 2 ;;
        -eval_data|--eval_data)     eval_data="$2"; shift 2 ;;
        -model|--model)             model="$2"; shift 2 ;;
        -name|--name)               run_name="$2"; shift 2 ;;
        -styles)                    styles=true; shift ;;
        -lora)                      lora=true; shift ;;
        -h|--help)                  usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

if [[ -z "${train_data:-}" || -z "${eval_data:-}" || -z "${model:-}" || -z "${run_name:-}" ]]; then
    echo "Error: -train_data, -eval_data, -model, and -name are required."
    usage
fi

CMD="phenogpt2_vision_training.py \
  -train_data ${train_data} \
  -eval_data ${eval_data} \
  -model ${model} \
  -name ${run_name}"

[[ -n "${styles:-}" ]] && CMD="${CMD} -styles"
[[ -n "${lora:-}" ]] && CMD="${CMD} -lora"
python ${CMD}

# -----------------------------------------------------------------------------
# Example sbatch submissions
# -----------------------------------------------------------------------------
# sbatch -p gpu-xe9680q --gres=gpu:h100:4 --cpus-per-gpu=2 --mem-per-cpu=50G \
#        --time=5-00:00:00 --profile=all --export=ALL --mail-type=ALL --mail-user=nguyenqm@chop.edu \
#        --wrap="bash finetuning_phenogpt2_vision.sh \
#                  -train_data /home/nguyenqm/projects/github/PhenoGPT2/vision_gmdb_data_new/phenotype_only \
#                  -eval_data /home/nguyenqm/projects/github/PhenoGPT2/vision_gmdb_data_new/phenotype_only \
#                  -name qwen3_8b_vl_phenotype_only -model Qwen/Qwen3-VL-8B-Instruct \
#                  -styles"

