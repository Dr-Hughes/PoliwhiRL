#!/bin/bash
set -e

Phase1() {
    # This phase takes us up until the point where a pokemon is received and the
    # player has gotten to the exit of the first town
    local PYTHON_CMD="python main.py"
    local BASE_EPOCHS=5
    local BASE_EPISODES=5001
    local BASE_SEQUENCE_LENGTH=4
    local BASE_BATCH_SIZE=256
    local EXTENDED_SEQUENCE_LENGTH=4
    local EXTENDED_BATCH_SIZE=512
    local EXTENDED_EPOCHS=10
    local EXTENDED_EPISODES=10001

    local total_start_time=$(date +%s)

    run_goal_learning() {
        local goals=$1
        local episode_length=$2
        local epochs=$3
        local num_episodes=$4
        local sequence_length=$5
        local erase=$6
        local batch_size=$7
        local early_stopping_avg_length=$((episode_length / 2))
        local continue_from_state=$8

        echo "Learning $goals goal(s)"
        local start_time=$(date +%s)
        $PYTHON_CMD --episode_length $episode_length --epochs $epochs --num_episodes $num_episodes \
                    --N_goals_target $goals --sequence_length $sequence_length --erase $erase --batch_size $batch_size \
                    --early_stopping_avg_length $early_stopping_avg_length --continue_from_state $continue_from_state
        local end_time=$(date +%s)
        local runtime=$((end_time - start_time))
        report_runtime "Runtime for Learning $goals goal(s)" $runtime
    }

    report_runtime() {
        local message=$1
        local seconds=$2
        local hours=$((seconds / 3600))
        local minutes=$(( (seconds % 3600) / 60 ))
        echo "$message: $hours hours $minutes minutes"
    }

    # Goal learning iterations
    run_goal_learning 1 50 $BASE_EPOCHS $BASE_EPISODES $BASE_SEQUENCE_LENGTH false $BASE_BATCH_SIZE false
    run_goal_learning 2 100 $BASE_EPOCHS $BASE_EPISODES $BASE_SEQUENCE_LENGTH false $BASE_BATCH_SIZE false
    run_goal_learning 3 500 $BASE_EPOCHS $BASE_EPISODES $BASE_SEQUENCE_LENGTH false $EXTENDED_BATCH_SIZE false
    run_goal_learning 4 500 $BASE_EPOCHS $BASE_EPISODES $EXTENDED_SEQUENCE_LENGTH false $EXTENDED_BATCH_SIZE false
    run_goal_learning 5 750 $EXTENDED_EPOCHS $EXTENDED_EPISODES $EXTENDED_SEQUENCE_LENGTH false $EXTENDED_BATCH_SIZE true
    run_goal_learning 6 1000 $EXTENDED_EPOCHS $EXTENDED_EPISODES $EXTENDED_SEQUENCE_LENGTH false $EXTENDED_BATCH_SIZE true
    run_goal_learning 7 1500 $EXTENDED_EPOCHS $EXTENDED_EPISODES $EXTENDED_SEQUENCE_LENGTH false $EXTENDED_BATCH_SIZE true

    # Report total runtime
    local total_end_time=$(date +%s)
    local total_runtime=$((total_end_time - total_start_time))
    report_runtime "Total runtime" $total_runtime
}

# Phase 2 will involve learning to battle and leveling up


# Initial learning phase
Phase1