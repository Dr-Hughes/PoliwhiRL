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
    input_dim = temp_env.get_game_area().shape
    output_dim = temp_env.action_space.n
    temp_env.close()
    del temp_env

    # Setup the PPO model
    device = config.get(
        "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Create ParallelPPO instance
    ppo = ParallelPPO(
        input_dims=input_dim,
        n_actions=output_dim,
        lr=config["learning_rate"],
        device=device,
        batch_size=config["batch_size"],
        gamma=config.get("gamma", 0.99),
        gae_lambda=config.get("gae_lambda", 0.95),
        policy_clip=config.get("policy_clip", 0.2),
        n_epochs=config.get("n_epochs", 10),
        value_coef=config.get("value_coef", 0.5),
        entropy_coef=config.get("entropy_coef", 0.01),
        max_grad_norm=config.get("max_grad_norm", 0.5),
        lr_decay_step=config.get("lr_decay_step", 1000),
        lr_decay_gamma=config.get("lr_decay_gamma", 0.9),
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
