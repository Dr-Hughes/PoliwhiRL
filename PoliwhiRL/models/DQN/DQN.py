# -*- coding: utf-8 -*-
import torch
from PoliwhiRL.environment import PyBoyEnvironment as Env

from PoliwhiRL.utils.utils import plot_metrics
from PoliwhiRL.models.DQN.agent import PokemonAgent


def setup_and_train(config):
    env = Env(config)
    state_shape = (
        env.get_screen_size() if config["vision"] else env.get_game_area().shape
    )
    num_actions = env.action_space.n

    agent = PokemonAgent(state_shape, num_actions, config, env)

    model_path = config["checkpoint"]
    try:
        agent.load_model(model_path)
        print(f"Loaded model from {model_path}")
        agent.optimizer = torch.optim.Adam(
            agent.model.parameters(), lr=config["learning_rate"]
        )
    except FileNotFoundError:
        print("No model found, training from scratch.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Training from scratch.")

    num_episodes = config["num_episodes"]
    episode_rewards, losses, epsilons = agent.train_agent(num_episodes)

    # Final save
    agent.save_model(model_path)
    print(f"Saved model to {model_path}")

    plot_metrics(episode_rewards, losses, epsilons, f"{config['N_goals_target']}_end")

    return agent, episode_rewards, losses, epsilons
