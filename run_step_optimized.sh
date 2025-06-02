#!/bin/bash
# Script to run training with step-optimized settings
# This uses the aggressive reward configuration designed to minimize steps

echo "Starting step-optimized training..."
echo "This configuration aggressively penalizes steps to find optimal paths"
echo ""

# Run with step-optimized reward settings
python main.py \
    --vision false \
    --episode_length 100 \
    --num_episodes 100 \
    --ppo_update_frequency 256 \
    --N_goals_target 1 \
    --output_base_dir "step_optimized/" \
    --ppo_num_agents 20 \
    --ppo_iterations 50 \
    --punish_steps true \
    --break_on_goal true \
    --report_episode true \
    --use_curriculum false \
    --reward_config "configs/default_configs/reward_settings_step_optimized.json" \
    --ppo_learning_rate 0.0005 \
    --ppo_gamma 0.95

echo ""
echo "Step-optimized training complete!"
echo "Check the results in step_optimized/Results/"