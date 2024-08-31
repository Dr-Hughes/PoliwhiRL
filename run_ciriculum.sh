
# Learn 1 goal
echo "Learning 1 goal"
python main.py --num_episodes_to_sample 32 --num_sequences_per_episode 16 --episode_length 250 --num_episodes 500 --N_goals_target 1 --sequence_length 4 --erase true

# Learn 2
echo "Learning 2 goals"
python main.py --num_episodes_to_sample 32 --num_sequences_per_episode 16 --episode_length 500 --num_episodes 500 --N_goals_target 2 --sequence_length 8 --erase false

# Learn 4
echo "Learning 4 goals"
python main.py --num_episodes_to_sample 32 --num_sequences_per_episode 16 --episode_length 5000 --num_episodes 500 --N_goals_target 4 --sequence_length 32 --erase false

# Learn 5
echo "Learning 5 goals"
python main.py --num_episodes_to_sample 32 --num_sequences_per_episode 16 --episode_length 5000 --num_episodes 500 --N_goals_target 5 --sequence_length 64 --erase false
