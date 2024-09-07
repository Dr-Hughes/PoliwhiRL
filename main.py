# -*- coding: utf-8 -*-

from PoliwhiRL.models.DQN import setup_and_train as setup_and_train_dqn
from PoliwhiRL.explorer import memory_collector
from PoliwhiRL.reward_evaluator import evaluate_reward_system
import os
import shutil
import argparse
import json
import pprint


class StoreBooleanAction(argparse.Action):
    # Custom action to store boolean values from command line arguments
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values.lower() in ("yes", "true", "t", "1"))


def load_default_config():
    default_config_path = "./configs/default_config.json"
    if os.path.exists(default_config_path):
        with open(default_config_path, "r") as f:
            return json.load(f)
    return {}


def load_user_config(config_path):
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


def merge_configs(default_config, user_config):
    merged_config = default_config.copy()
    merged_config.update(user_config)
    return merged_config


def parse_args():
    default_config = load_default_config()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--use_config", type=str, default=None, help="Path to user config file"
    )

    (
        args,
        unknown,
    ) = (
        parser.parse_known_args()
    )  # Parse known args first to get config file if specified

    user_config = load_user_config(args.use_config)
    config = merge_configs(default_config, user_config)

    # Dynamically add other arguments based on the merged config
    for key, value in config.items():
        if isinstance(value, bool):
            parser.add_argument(
                f"--{key}", type=str, action=StoreBooleanAction, default=value
            )
        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)

    return vars(parser.parse_args())  # Return arguments as a dictionary


def main():
    config = parse_args()

    if config["erase"]:
        print("Erasing previous training outputs")
        shutil.rmtree("Training Outputs", ignore_errors=True)

    if config["verbose"]:
        pprint.pprint(config)
    if config["model"] in ["DQN"]:
        setup_and_train_dqn(config)
    elif config["model"] == "explore":
        memory_collector(config)
    elif config["model"] == "evaluate":
        evaluate_reward_system(config)
    else:
        raise ValueError(f"Model {config['model']} not recognized")


if __name__ == "__main__":
    main()
