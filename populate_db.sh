#!/bin/bash

# Check if the number of runs is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <number_of_runs>"
    exit 1
fi

# Number of runs
N=$1

# Function to run the Python script
run_python_script() {
    python main.py --use_config configs/epic_adventure_config.json  --erase false  --checkpoint Run_$1 --results_dir Results/Run_$1 --record_path Run_$1 --export_state_loc States/Run_$1
}

# Export the function so GNU Parallel can use it
export -f run_python_script

# Use GNU Parallel to run N instances of the script
parallel -j$N run_python_script ::: $(seq 1 $N)