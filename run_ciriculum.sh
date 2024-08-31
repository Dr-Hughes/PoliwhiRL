
# Learn 1 goal 
echo "Learning 1 goal"
python main.py --num_episodes_to_sample 32 --num_sequences_per_episode 16 --episode_length 250 --N_goals_target 1 --num_episodes 500 --N_goals_target 1 --sequence_length 4

# Learn 2 
echo "Learning 2 goals"
python main.py --num_episodes_to_sample 32 --num_sequences_per_episode 16 --episode_length 250 --N_goals_target 1 --num_episodes 500 --N_goals_target 2 --sequence_length 8

# Learn 4 
echo "Learning 4 goals"
python main.py --num_episodes_to_sample 32 --num_sequences_per_episode 16 --episode_length 250 --N_goals_target 1 --num_episodes 500 --N_goals_target 4 --sequence_length 32

# Learn 5 
echo "Learning 5 goals"
python main.py --num_episodes_to_sample 32 --num_sequences_per_episode 16 --episode_length 250 --N_goals_target 1 --num_episodes 500 --N_goals_target 5 --sequence_length 64
