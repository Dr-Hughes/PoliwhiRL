#!/bin/bash
set -e

train_pokemon_agent() {
    local PYTHON_CMD="python main.py"
    local EPOCHS=10
    local EPISODES=1000
    local SEQUENCE_LENGTH=6
    local BATCH_SIZE=64
    local GOALS=6
    local EPISODE_LENGTH=50
    local EARLY_STOPPING_AVG_LENGTH=$((EPISODE_LENGTH / 2))

    local total_start_time=$(date +%s)

    echo "Starting Pokemon Agent Training"
    echo "Goals: $GOALS, Episode Length: $EPISODE_LENGTH, Epochs: $EPOCHS, Episodes: $EPISODES"

    $PYTHON_CMD --episode_length $EPISODE_LENGTH \
                --epochs $EPOCHS \
                --num_episodes $EPISODES \
                --N_goals_target $GOALS \
                --sequence_length $SEQUENCE_LENGTH \
                --batch_size $BATCH_SIZE \
                --early_stopping_avg_length $EARLY_STOPPING_AVG_LENGTH \
                --erase true \
                --continue_from_state "false" \
                --continue_from_state_loc "None"

    local total_end_time=$(date +%s)
    local total_runtime=$((total_end_time - total_start_time))

    report_runtime "Total training runtime" $total_runtime
}

report_runtime() {
    local message=$1
    local seconds=$2
    local hours=$((seconds / 3600))
    local minutes=$(( (seconds % 3600) / 60 ))
    echo "$message: $hours hours $minutes minutes"
}

# Run the training
train_pokemon_agent