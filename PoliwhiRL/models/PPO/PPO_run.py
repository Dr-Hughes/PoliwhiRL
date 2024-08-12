# -*- coding: utf-8 -*-
from .PPO import ParallelPPO  # Import the modified ParallelPPO class
from PoliwhiRL.environment import PyBoyEnvironment as Env
import torch
import multiprocessing as mp


def create_env(config):
    return Env(config)


def setup_and_train_ppo(config):
    # Create a temporary environment to get input dimensions and action space
    temp_env = create_env(config)
    vision = config.get("vision", False)
    if vision:
        height, width, channels = temp_env.get_screen_size()
        input_dim = (channels, height, width)  # PyTorch expects channels first
    else:
        input_dim = temp_env.get_game_area().shape
    output_dim = temp_env.action_space.n
    temp_env.close()
    del temp_env


    # Create ParallelPPO instance
    ppo = ParallelPPO(
        input_dims=input_dim,
        n_actions=output_dim,
        config=config
    )

    # Load pre-trained models if needed
    ppo.load_models()

    # Set the start method for multiprocessing
    mp.set_start_method("spawn", force=True)

    # Start parallel training
    trained_ppo = ppo.train_parallel(create_env, config)

    # Save the model
    trained_ppo.save_models()

    return trained_ppo
