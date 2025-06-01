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
    100    # 1 goal - reduced from 150
    150    # 2 goals - reduced from 200
    200    # 3 goals - reduced from 300
    250    # 4 goals - reduced from 400
    350    # 5 goals - reduced from 600
    500    # 6 goals - reduced from 1000
    700    # 7 goals - reduced from 1500
)

for goals in $(seq $start_goals $max_goals); do
    echo "Running with $goals goals"

    # Get the episode length from the array (arrays are 0-indexed)
    episode_length=${episode_lengths[$((goals-1))]}

    # Set checkpoint path for loading (except for first stage)
    if [ $goals -eq 1 ]; then
        load_checkpoint=""
        echo "Starting from scratch for stage 1"
    else
        prev_goals=$((goals - 1))
        load_checkpoint="./stage_${prev_goals}/Checkpoints"
        echo "Loading checkpoint from: $load_checkpoint"
    fi

    # Run the main.py with the calculated parameters using memory-based multi-agent
    python main.py \
        --vision false \
        --episode_length $episode_length \
        --num_episodes 10 \
        --ppo_update_frequency 256 \
        --N_goals_target $goals \
        --output_base_dir "stage_${goals}/" \
        --ppo_num_agents 20 \
        --ppo_iterations 10 \
        --punish_steps true \
        --report_episode false \
        --use_curriculum false \
        --break_on_goal true \
        --load_checkpoint "$load_checkpoint"

    # Prepare for the next stage if there is one
    next_goals=$((goals + 1))
    if [ $next_goals -le $max_goals ]; then
        echo "Preparing for $next_goals goals"
        mkdir -p "stage_$next_goals"
        cp -r "./stage_${goals}/Checkpoints" "./stage_${next_goals}"
    fi
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

        # Load checkpoint from the previous run
        load_checkpoint="./stage_${goals}/Checkpoints"

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
