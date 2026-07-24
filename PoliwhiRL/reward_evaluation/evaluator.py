# -*- coding: utf-8 -*-
import numpy as np
from PoliwhiRL.environment import PyBoyEnvironment as Env
from .moves import button_presses
from tqdm import tqdm


def evaluate_reward_system(config):
    env = Env(config)
    try:
        output_path = config["results_dir"]

        print(f"Evaluating reward system with {len(button_presses)} button presses.")

        # Reset the environment
        env.reset()
        rewards = []
        # Apply each action and observe the result
        for action in tqdm(button_presses):
            observation, reward, done, info = env.step(action)
            rewards.append(reward)
            env.save_debug_step_img_data("evaluation", output_path)
            if done:
                print(
                    "Environment signalled completion before all actions were executed."
                )
                break

        summary_path = env.finalize_debug_run("evaluation", output_path)
        rc = env.reward_calculator
        print(f"Total reward: {np.sum(rewards)}")
        # Per-source breakdown + goal success — sanity-check a new reward
        # design offline: confirm milestones pay, battle reward is
        # small/capped, intrinsic novelty behaves, without full training.
        print("Per-source reward breakdown:")
        for src, val in sorted(rc.get_episode_breakdown().items()):
            print(f"  {src:20s} {val:10.2f}")
        print(
            f"Goal success (all thresholds met): {rc.goals.all_goal_thresholds_met()}"
        )
        print(f"Run summary: {summary_path}")
        print("Evaluation complete.")

        return rewards
    finally:
        # Ensure environment is properly closed
        env.close()
