#!/bin/bash
# Curriculum learning script optimized for step minimization
# This script progressively trains the agent with tighter step constraints

# Define the range for N_goals_target
start_goals=1
max_goals=7  # 7 total goals

# Optional: Run optimization passes with progressively tighter constraints
run_optimization_pass=true
optimization_reduction=0.8  # Reduce episode length by 20% in optimization pass

# Define a list of episode lengths for each goal level (aggressive step minimization)
# These are much tighter to force the agent to find optimal paths
episode_lengths=(
    75    # 1 goal - reduced from 150
    150    # 2 goals - reduced from 200
    600    # 3 goals - reduced from 300
    1000    # 4 goals - reduced from 400
    2000    # 5 goals - reduced from 600
    2500    # 6 goals - reduced from 1000
    5000    # 7 goals - reduced from 1500
)

for goals in $(seq $start_goals $max_goals); do
    echo "Running with $goals goals"

    # Get the episode length from the array (arrays are 0-indexed)
    episode_length=${episode_lengths[$((goals-1))]}

    # Copy checkpoint from previous stage (except for first stage)
    if [ $goals -eq 1 ]; then
        echo "Starting from scratch for stage 1"
        load_checkpoint=""
    else
        prev_goals=$((goals - 1))
        echo "Copying checkpoint from stage ${prev_goals} to stage ${goals}"
        mkdir -p "./stage_${goals}"
        if [ -d "./stage_${prev_goals}/Checkpoints" ]; then
            cp -r "./stage_${prev_goals}/Checkpoints" "./stage_${goals}/Checkpoints"
        fi
        # Load from current stage's checkpoint directory
        load_checkpoint="./stage_${goals}/Checkpoints"
    fi

    # Run the main.py with the calculated parameters using memory-based multi-agent
    python main.py \
        --vision false \
        --episode_length $episode_length \
        --num_episodes 20 \
        --ppo_update_frequency 256 \
        --N_goals_target $goals \
        --output_base_dir "stage_${goals}/" \
        --ppo_num_agents 20 \
        --ppo_iterations 11 \
        --punish_steps true \
        --report_episode false \
        --use_curriculum false \
        --break_on_goal true \
        --load_checkpoint "$load_checkpoint"
done

# Optional optimization pass: Further reduce episode lengths for already-trained goals
if [ "$run_optimization_pass" = true ]; then
    echo -e "\n\n=== STARTING OPTIMIZATION PASS ===\n"
    echo "Running optimization pass to further minimize steps..."

    for goals in $(seq $start_goals $max_goals); do
        echo -e "\nOptimizing stage $goals with reduced episode length"

        # Calculate reduced episode length
        original_length=${episode_lengths[$((goals-1))]}
        optimized_length=$(echo "$original_length * $optimization_reduction" | bc | cut -d. -f1)

        # Copy checkpoint from regular training to optimized directory
        echo "Copying checkpoint from stage ${goals} to stage ${goals}_optimized"
        mkdir -p "./stage_${goals}_optimized"
        if [ -d "./stage_${goals}/Checkpoints" ]; then
            cp -r "./stage_${goals}/Checkpoints" "./stage_${goals}_optimized/Checkpoints"
        fi

        # Load checkpoint from optimized directory
        load_checkpoint="./stage_${goals}_optimized/Checkpoints"

        # Run optimization with tighter constraints using memory-based multi-agent
        python main.py \
            --vision false \
            --episode_length $optimized_length \
            --num_episodes 30 \
            --ppo_update_frequency 256 \
            --N_goals_target $goals \
            --output_base_dir "stage_${goals}_optimized/" \
            --ppo_num_agents 20 \
            --ppo_iterations 30 \
            --punish_steps true \
            --report_episode false \
            --use_curriculum false \
            --break_on_goal true \
            --load_checkpoint "$load_checkpoint"

        echo "Optimization for $goals goals complete (reduced from $original_length to $optimized_length steps)"
    done

    echo -e "\n=== OPTIMIZATION PASS COMPLETE ===\n"
fi
