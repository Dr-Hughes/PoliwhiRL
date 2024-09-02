# -*- coding: utf-8 -*-
import torch
from PoliwhiRL.environment import PyBoyEnvironment as Env
from PoliwhiRL.models.DQN.agent import PokemonAgent
from  PoliwhiRL.models.DQN.multi_agent import setup_and_train_multi_agent


def setup_and_train(config):
    env = Env(config)
    state_shape = (
        env.get_screen_size() if config["vision"] else env.get_game_area().shape
    )
    num_actions = env.action_space.n

    #agent = PokemonAgent(state_shape, num_actions, config, env)
    agent = setup_and_train_multi_agent(state_shape, num_actions, config)

    # model_path = config["checkpoint"]
    # try:
    #     agent.load_model(model_path)
    #     print(f"Loaded model from {model_path}")
    #     agent.optimizer = torch.optim.Adam(
    #         agent.model.parameters(), lr=config["learning_rate"]
    #     )
    # except FileNotFoundError:
    #     print("No model found, training from scratch.")
    # except Exception as e:
    #     print(f"Error loading model: {e}")
    #     print("Training from scratch.")

    # num_episodes = config["num_episodes"]
    # agent.train_agent(num_episodes)

    # # Final save
    # agent.save_model(model_path)
    # print(f"Saved model to {model_path}")
