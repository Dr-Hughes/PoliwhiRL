# -*- coding: utf-8 -*-
from .training_functions import (
    setup_environment_and_model,
    train,
    save_checkpoint,
)


def setup_and_train_ppo(config):

    updated_vars = {}
    for var in ["episode_length", "num_episodes", "N_goals_target"]:
        # check if var is a string
        if isinstance(config[var], str) and "," in config[var]:
            updated_vars[var] = [int(i) for i in config[var].split(",")]
        else:
            updated_vars[var] = [int(config[var])]

    for idx, v in enumerate(updated_vars["episode_length"]):
        config["episode_length"] = updated_vars["episode_length"][idx]
        config["num_episodes"] = updated_vars["num_episodes"][idx]
        config["N_goals_target"] = updated_vars["N_goals_target"][idx]

        env, model, optimizer, scheduler, start_episode = setup_environment_and_model(
            config
        )
        train(
            model, env, optimizer, scheduler, config, start_episode
        )  # Train the model
        save_checkpoint(
            model,
            config["checkpoint"],
            start_episode + config["num_episodes"],
        )
        env.close()
