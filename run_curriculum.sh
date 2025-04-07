# Define the range for N_goals_target
start_goals=1
max_goals=7  # Set your maximum number of goals here

for goals in $(seq $start_goals $max_goals); do
    echo "Running with $goals goals"

    # Calculate episode length based on number of goals
    episode_length=$((100 * goals))

    # Run the main.py with the calculated parameters
    python main.py \
        --vision false \
        --episode_length $episode_length \
        --num_episodes 20 \
        --ppo_update_frequency 256 \
        --N_goals_target $goals \
        --output_base_dir "stage_${goals}/" \
        --ppo_num_agents 20 \
        --ppo_iterations 20 \
        --punish_steps true \
        --report_episode false \
        --use_curriculum false \

    # Prepare for the next stage if there is one
    next_goals=$((goals + 1))
    if [ $next_goals -le $max_goals ]; then
        echo "Preparing for $next_goals goals"
        mkdir -p "stage_$next_goals"
        cp -r "./stage_${goals}/Checkpoints" "./stage_${next_goals}"
    fi
done