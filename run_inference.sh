#!/bin/bash
# =============================================================================
# run_inference.sh
# PhenoGPT2 — vLLM inference launcher
#
# Usage:
#   bash run_inference.sh -i <input> -o <output_dir> [options]
#
# Example (SLURM):
#   sbatch -p gpuq --gres=gpu:a100:1 --cpus-per-gpu=2 --mem-per-cpu=50G \
#          --time=5-00:00:00 --export=ALL \
#          --wrap="bash run_inference.sh \
#                  -i ./data/example/text_examples.json \
#                  -o ./example_testing \
#                  -model_dir ./models/phenogpt2 \
#                  -negation_model Qwen/Qwen3-4B-Instruct-2507 \
#                  -batch_size 7 -chunk_batch_size 7 -index 0 \
#                  -wc 0 -text_only -negation"
# =============================================================================

# --- show usage ---
usage() {
    echo "Usage: $0 -i <input_file> -o <output_dir> \\"
    echo "          [-model_dir <path>] [-index <n>] \\"
    echo "          [-negation_model <name>] [-batch_size <n>] \\"
    echo "          [-chunk_batch_size <n>] [-wc <word_count>] \\"
    echo "          [-vision_dir <str>] \\"
    echo "          [-lora] [-negation] [-text_only] [-vision_only]"
    exit 1
}

# --- defaults ---
wc=0
lora=false
negation_enabled=false
vision_dir=""
text_only=false
vision_only=false

# --- parse arguments ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -i|--input)              input="$2";              shift 2 ;;
        -o|--output)             output="$2";             shift 2 ;;
        -model_dir|--model_dir)  model_dir="$2";          shift 2 ;;
        -negation_model|--negation_model) negation_model="$2"; shift 2 ;;
        -attn_implementation|--attn_implementation) attn_implementation="$2"; shift 2 ;;
        -batch_size|--batch_size)       batch_size="$2";       shift 2 ;;
        -chunk_batch_size|--chunk_batch_size) chunk_batch_size="$2"; shift 2 ;;
        -index|--index)          index="$2";              shift 2 ;;
        -vision_dir|--vision_dir)        vision_dir="$2";             shift 2 ;;
        -wc|--wc)                wc="$2";                 shift 2 ;;
        -lora|--lora)            lora=true;               shift   ;;
        -negation|--negation)    negation_enabled=true;   shift   ;;
        -text_only|--text_only)  text_only=true;          shift   ;;
        -vision_only|--vision_only) vision_only=true;     shift   ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
done

# --- validate required args ---
if [[ -z "$input" || -z "$output" ]]; then
    echo "Error: -i and -o are required."
    usage
fi

# --- build command ---
cmd="python inference.py -i \"$input\" -o \"$output\""

[[ -n "$model_dir" ]]            && cmd+=" -model_dir \"$model_dir\""
[[ -n "$negation_model" ]]       && cmd+=" -negation_model \"$negation_model\""
[[ -n "$attn_implementation" ]]  && cmd+=" -attn_implementation \"$attn_implementation\""
[[ -n "$batch_size" ]]           && cmd+=" -batch_size \"$batch_size\""
[[ -n "$chunk_batch_size" ]]     && cmd+=" -chunk_batch_size \"$chunk_batch_size\""
[[ -n "$index" ]]                && cmd+=" -index \"$index\""
[[ -n "$vision_dir" ]]               && cmd+=" -vision_dir \"$vision_dir\""
[[ "$wc" -gt 0 ]]                && cmd+=" -wc $wc"
[[ "$lora"             == true ]] && cmd+=" -lora"
[[ "$negation_enabled" == true ]] && cmd+=" -negation"
[[ "$text_only"        == true ]] && cmd+=" --text_only"
[[ "$vision_only"      == true ]] && cmd+=" --vision_only"

echo "Executing: $cmd"
eval $cmd


# sbatch -p gpu-xe9680q --gres=gpu:h100:1 --cpus-per-gpu=1 --mem-per-cpu=50G --profile=all \
# --time=2-00:00:00 --export=ALL \
# --wrap="bash run_inference.sh \
# -i /home/nguyenqm/projects/PhenoGPT2/testing/phenotagger/input/ID-68/ \
# -o /home/nguyenqm/projects/github_official/PhenoGPT2_vllm/example_testing/ID-68 \
# -model_dir /home/nguyenqm/projects/github/PhenoGPT2/phenogpt2_qwen3_ehr_8b_ft_nofilter/model -negation_model "Qwen/Qwen3-4B-Instruct-2507" \
# -attn_implementation flash_attention_2 -index 0 -wc 0 -text_only -negation"