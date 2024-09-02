import torch
import torch.multiprocessing as mp
from PoliwhiRL.environment import PyBoyEnvironment as Env
from PoliwhiRL.models.DQN.DQNModel import TransformerDQN
import os

from PoliwhiRL.replay.sequence_storage import SequenceStorage

def init_shared_model(state_shape, action_size, config):
    device = torch.device(config["device"])
    shared_model = TransformerDQN(state_shape, action_size).to(device)
    shared_model.share_memory()  # Required for multiprocessing
    
    optimizer = torch.optim.Adam(
        shared_model.parameters(),
        lr=config["learning_rate"]
    )
    
    return shared_model, optimizer, device

def gather_experiences(shared_model, config, state_shape, action_size, num_agents, num_episodes):
    episodes_per_update = num_agents  # Run all agents for one episode before updating
    total_episodes = 0
    experiences = []
    while total_episodes < num_episodes:
        with mp.Pool(processes=num_agents) as pool:
            # Run episodes in parallel
            results = pool.starmap(run_episode, [(shared_model, config, state_shape, action_size, episode_num) 
                                                 for episode_num in range(episodes_per_update)])
                
            # Collect results
            for episode_experiences in results:
                experiences.extend(episode_experiences)
        total_episodes += episodes_per_update
        print(f"Completed {total_episodes} episodes")
    return experiences

def run_episode(shared_model, config, state_shape, action_size, episode_num):
    env = Env(config)
    state = env.reset()
    done = False
    episode_reward = 0
    episode_experiences = []
    
    while not done:
        # Select action
        action = get_action(shared_model, state, config["device"])
        
        # Take action in environment
        next_state, reward, done, _ = env.step(action)
        
        # Store experience
        episode_experiences.append((state, action, reward, next_state, done))
        
        state = next_state
        episode_reward += reward
    
    print(f"Episode {episode_num}, Reward: {episode_reward}")
    return episode_experiences

def get_action(model, state, device):
    state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model(state)
    return q_values[0, -1, :].argmax().item()

def train_step(shared_model, optimizer, shared_replay_buffer, config, device):
    if len(shared_replay_buffer) < config["batch_size"]:
        return
    
    batch = shared_replay_buffer.sample(config["batch_size"])
    if batch is None:
        return
    
    states, actions, rewards, next_states, dones, sequence_ids, weights = batch
    
    # Move everything to the correct device
    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)
    weights = weights.to(device)
    
    # Compute Q-values
    current_q_values = shared_model(states).gather(2, actions.unsqueeze(-1)).squeeze(-1)
    
    with torch.no_grad():
        next_q_values = shared_model(next_states).max(2)[0]
        target_q_values = rewards + config["gamma"] * next_q_values * (~dones)
    
    # Compute loss
    loss = torch.nn.functional.smooth_l1_loss(current_q_values, target_q_values, reduction='none')
    loss = (loss.mean(dim=1) * weights).mean()
    
    # Optimize
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(shared_model.parameters(), max_norm=10.0)
    optimizer.step()
    
    # Update priorities in the replay buffer
    td_errors = torch.abs(current_q_values - target_q_values).mean(dim=1).detach().cpu().numpy()
    shared_replay_buffer.update_priorities(sequence_ids, td_errors)

def setup_and_train_multi_agent(state_shape, action_size, config, num_agents=4):
    os.makedirs(os.path.dirname(config["db_path"]), exist_ok=True)
    shared_replay_buffer = SequenceStorage(config["db_path"], 
                                           config["replay_buffer_capacity"], 
                                           config["sequence_length"])
    shared_model, optimizer, device = init_shared_model(state_shape, action_size, config)

    for episode in range(2):
        experiences = gather_experiences(shared_model, config, state_shape, action_size, num_agents, 1)
        for experience in experiences:
            shared_replay_buffer.add(*experience)
        
        # Perform training steps
        for _ in range(config["epochs"]):
            train_step(shared_model, optimizer, shared_replay_buffer, config, device)

    return shared_model, optimizer