# Define the range for N_goals_target
start_goals=1
max_goals=7  # Set your maximum number of goals here

# Define a list of episode lengths for each goal level
episode_lengths=(
    100    # 1 goal
    200   # 2 goals
    300   # 3 goals
    500   # 4 goals
    2500   # 5 goals
    3000   # 6 goals
    3500   # 7 goals
)

for goals in $(seq $start_goals $max_goals); do
    echo "Running with $goals goals"

    # Get the episode length from the array (arrays are 0-indexed)
    episode_length=${episode_lengths[$((goals-1))]}

    # Run the main.py with the calculated parameters
    python main.py \
        --vision false \
        --episode_length $episode_length \
        --num_episodes 50 \
        --ppo_update_frequency 256 \
        --N_goals_target $goals \
        --output_base_dir "stage_${goals}/" \
        --ppo_num_agents 20 \
        --ppo_iterations 50 \
        --punish_steps true \
        --report_episode false \
        --use_curriculum false

    # Prepare for the next stage if there is one
    next_goals=$((goals + 1))
    if [ $next_goals -le $max_goals ]; then
        echo "Preparing for $next_goals goals"
        mkdir -p "stage_$next_goals"
        cp -r "./stage_${goals}/Checkpoints" "./stage_${next_goals}"
    fi
done