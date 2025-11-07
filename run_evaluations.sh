#!/bin/bash
# Display usage if the required arguments are not provided
usage() {
    echo "Usage: $0 [-model_dir <model_dir>] -result_dir <result_dir>] -name <name> [-negation]"
    exit 1
}

# Parse the command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -model_dir)
            if [ -z "$2" ]; then echo "Error: -model_dir requires a value."; usage; fi
            model_dir="$2"; shift ;;
        -result_dir)
            if [ -z "$2" ]; then echo "Warning: -result_dir is missing, using model_dir instead."; usage; fi
            result_dir="$2"; shift ;;
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
if [[ -n "$result_dir" ]]; then
    cmd+=" -result_dir $result_dir"
fi
if [[ -n "$negation" ]]; then
    cmd+=" -negation"
fi
# Run the python command
echo "Executing: $cmd"
eval $cmd
