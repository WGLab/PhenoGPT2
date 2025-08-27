#!/bin/bash

# Load CUDA module
module load CUDA/12.1.1

# Show usage message
usage() {
    echo "Usage: $0 -i <input_file> -o <output_dir> [-model_dir <model_directory>] [-index <index>] [-text_only] [-vision_only] [-vision <llava-med|llama-vision>] [-wc <word_count>] [-lora] [-negation]"
    exit 1
}

# Initialize default values
wc=0
lora=false
negation_enabled=true
vision=""
text_only=false
vision_only=false

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -i|--input)
            input="$2"
            shift 2
            ;;
        -o|--output)
            output="$2"
            shift 2
            ;;
        -model_dir|--model_dir)
            model_dir="$2"
            shift 2
            ;;
        -index|--index)
            index="$2"
            shift 2
            ;;
        -vision|--vision)
            vision="$2"
            shift 2
            ;;
        -wc|--wc)
            wc="$2"
            shift 2
            ;;
        -lora|--lora)
            lora=true
            shift
            ;;
        -negation)
            negation_enabled=false
            shift
            ;;
        -text_only)
            text_only=true
            shift
            ;;
        -vision_only)
            vision_only=true
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            usage
            ;;
    esac
done

# Check required arguments
if [[ -z "$input" || -z "$output" ]]; then
    echo "Error: -i and -o are required."
    usage
fi

# Construct Python command
cmd="python inference.py -i \"$input\" -o \"$output\""

# Append optional arguments
[[ -n "$model_dir" ]] && cmd+=" -model_dir \"$model_dir\""
[[ -n "$index" ]] && cmd+=" -index \"$index\""
[[ -n "$vision" ]] && cmd+=" -vision \"$vision\""
[[ "$wc" -gt 0 ]] && cmd+=" -wc $wc"
[[ "$lora" == true ]] && cmd+=" -lora"
[[ "$negation_enabled" == false ]] && cmd+=" -negation"
[[ "$text_only" == true ]] && cmd+=" --text_only"
[[ "$vision_only" == true ]] && cmd+=" --vision_only"

# Show and execute
echo "Executing: $cmd"
eval $cmd

# sbatch -p gpuq --gres=gpu:a100:1 \
# --cpus-per-gpu=3 --mem-per-cpu=50G \
# --profile=all --time=5-00:00:00 --export=ALL \
# --mail-type=ALL --mail-user= \
# --wrap="bash run_inference.sh -i ./data/example/vision_examples.json \
#         -o vision_example \
#         -model_dir /home/nguyenqm/projects/github/PhenoGPT2/phenogpt2_EHR_L318B_text_FPFF/model/ \
#         -vision /home/nguyenqm/projects/github/PhenoGPT2/phenogpt2_L323BVision_LORA_fp80_fc10_bc5_nc5_with_styles/ \
#         -index 0 -negation -wc 0 -vision_only"

