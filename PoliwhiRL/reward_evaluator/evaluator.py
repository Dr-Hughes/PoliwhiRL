# -*- coding: utf-8 -*-
import numpy as np
from PoliwhiRL.environment import PyBoyEnvironment as Env
from .moves import button_presses
from tqdm import tqdm


def evaluate_reward_system(config):
    env = Env(config)

    output_path = config["results_dir"]

    print(f"Evaluating reward system with {len(button_presses)} button presses.")

    # Reset the environment
    env.reset()
    rewards = []
    # Apply each action and observe the result
    for action in tqdm(button_presses):
        observation, reward, done, info = env.step(action)
        rewards.append(reward)
        env.save_step_img_data("evaluation", output_path)
        if done:
            print("Environment signalled completion before all actions were executed.")
            break

    print(f"Total reward: {np.sum(rewards)}")
    print("Evaluation complete.")

    return rewards
