#!/bin/bash
module load CUDA/12.1.1
# Display usage if the required arguments are not provided
usage() {
    echo "Usage: $0 [-model_dir <model_dir>] -name <name> [-negation]"
    exit 1
}

# Parse the command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -model_dir)
            if [ -z "$2" ]; then echo "Error: -model_dir requires a value."; usage; fi
            model_dir="$2"; shift ;;
        -name)
            if [ -z "$2" ]; then echo "Error: -name requires a value."; usage; fi
            name="$2"; shift ;;
        -negation) negation=true ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Check if all required arguments are provided
if [[ -z "$model_dir" || -z "$name" ]]; then
    echo "Error: -model_dir, and -name are all required."
    usage
fi

# Run python script
# Construct the python command dynamically
cmd="python ./scripts/evaluations.py -model_dir \"$model_dir\" -name \"$name\""
if [[ -n "$negation" ]]; then
    cmd+=" -negation"
fi
# Run the python command
echo "Executing: $cmd"
eval $cmd
# sbatch -p gpu-xe9680q --gres=gpu:h100:1 --cpus-per-gpu=3 --mem-per-cpu=50G --profile=all --time=5-00:00:00 --export=ALL --mail-type=ALL --mail-user=nguyenqm@chop.edu --wrap="bash run_evaluations.sh -model_dir /home/nguyenqm/projects/github/PhenoGPT2/phenogpt2_L318B_text_FPLoRA_new/ -name Arcus_384split_new -negation"
