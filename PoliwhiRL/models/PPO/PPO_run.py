# -*- coding: utf-8 -*-
from .training_functions import (
    setup_environment_and_model,
    train,
)


def setup_and_train_ppo(config):

    env, model, start_episode = setup_environment_and_model(config)
    mdl = train(model, env, config, start_episode)
    env.close()

    mdl.save_models()
