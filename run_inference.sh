#!/bin/bash
#SBATCH --job-name=phenogpt2_inference
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=3
#SBATCH --mem=400G
#SBATCH --mem-per-cpu=100G
#SBATCH --time=1-00:00:00
#SBATCH --export=ALL
#SBATCH --profile=all
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
module load CUDA/12.1.1
# Display usage if the required arguments are not provided
usage() {
    echo "Usage: $0 -i <input_file> -o <output_file> [-model_dir <model_directory>]"
    exit 1
}

# Parse the command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -i) input="$2"; shift ;;
        -o) output="$2"; shift ;;
        -model_dir) model_dir="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Check if required arguments are provided
if [[ -z "$input" || -z "$output" ]]; then
    echo "Error: -i and -o are required arguments."
    usage
fi

# Run the Python script with the provided arguments
if [[ -z "$model_dir" ]]; then
    # If -model_dir is not provided, run without it
    python inference.py -i "$input" -o "$output"
else
    # If -model_dir is provided, include it in the command
    python inference.py -i "$input" -o "$output" -model_dir "$model_dir"
fi
#sbatch -p gpuq --gres=gpu:a100:1 --cpus-per-gpu=3 --mem-per-cpu=50G --profile=all --export=ALL --wrap="bash run_inference.sh -i testing/input/ -o testing/output/"