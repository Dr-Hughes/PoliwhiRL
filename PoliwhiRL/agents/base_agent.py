# -*- coding: utf-8 -*-
from collections import deque
from PoliwhiRL.environment import PyBoyEnvironment as Env
import numpy as np


class BaseAgent:
    def __init__(self):
        pass

    def step(self, env, state_sequence, model):
        action = self.get_action(model, state_sequence)
        next_state, reward, done, _ = env.step(action)
        return action, reward, next_state, done

    def get_action(self, model, state_sequence):
        return np.random.randint(0, model.action_size)

    def run_episode(
        self,
        model,
        config,
        record_loc=None,
        load_path=None,
        save_path=None,
    ):
        env = Env(config)
        try:
            if load_path is not None:
                state = env.load_gym_state(
                    load_path, config["episode_length"], config["N_goals_target"]
                )
            else:
                state = env.reset()

            if record_loc is not None:
                env.enable_record(record_loc, False)
            sequence_length = config["sequence_length"]
            done = False
            episode_experiences = []

            state_sequence = deque([state], maxlen=sequence_length)

            while not done:
                action, reward, next_state, done = self.step(
                    env, list(state_sequence), model
                )
                episode_experiences.append((state, action, reward, next_state, done))
                state_sequence.append(next_state)
                state = next_state

            if save_path is not None:
                env.save_gym_state(save_path)
            steps = env.steps
            return episode_experiences, steps
        finally:
            # Ensure environment is properly closed
            env.close()
