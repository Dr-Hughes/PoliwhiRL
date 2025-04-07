# -*- coding: utf-8 -*-

from PoliwhiRL import setup_and_train_DQN, setup_and_train_PPO
from PoliwhiRL.explorer import memory_collector
from PoliwhiRL.reward_evaluator import evaluate_reward_system
import os
import shutil
import argparse
import json
import pprint
from glob import glob


class StoreBooleanAction(argparse.Action):
    # Custom action to store boolean values from command line arguments
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values.lower() in ("yes", "true", "t", "1"))


def load_default_config():
    default_config_path = "./configs/default_configs"
    jsons = glob(default_config_path + "/*.json")
    files_to_concat = []
    if len(jsons) == 0:
        raise FileNotFoundError("No default config files found")
    for j in jsons:
        with open(j, "r") as f:
            settings = json.load(f)
            if "outputs" in j:
                modify_outputs_settings(settings)
            files_to_concat.append(settings)
    default_config = {}
    for file in files_to_concat:
        default_config.update(file)
    return default_config


def modify_outputs_settings(settings):
    for k, v in settings.items():
        if k != "output_base_dir":
            settings[k] = settings["output_base_dir"] + v


def load_user_config(config_path):
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


def merge_configs(default_config, user_config):
    merged_config = default_config.copy()
    merged_config.update(user_config)

    if merged_config["output_base_dir"] != default_config["output_base_dir"]:
        for k, v in merged_config.items():
            if isinstance(v, (str)) and default_config["output_base_dir"] in v:
                merged_config[k] = v.replace(
                    default_config["output_base_dir"], merged_config["output_base_dir"]
                )

    return merged_config


def parse_args():
    # Step 1: Load default config
    default_config = load_default_config()

    # Step 2: Parse just to get the config file path
    initial_parser = argparse.ArgumentParser()
    initial_parser.add_argument(
        "--use_config", type=str, default=None, help="Path to user config file"
    )
    initial_args, _ = initial_parser.parse_known_args()

    # Step 3: Load user config if specified
    user_config = load_user_config(initial_args.use_config)

    # Step 4: Merge default and user configs
    merged_config = merge_configs(default_config, user_config)

    # Step 5: Create a parser for all possible arguments
    # We'll use this to parse and type-convert the command line args
    cmd_parser = argparse.ArgumentParser()

    # Add all config keys as possible command line arguments
    for key, value in merged_config.items():
        if isinstance(value, bool):
            cmd_parser.add_argument(
                f"--{key}",
                type=str,
                action=StoreBooleanAction,
                # Don't set defaults here - we'll handle priority manually
                default=argparse.SUPPRESS,
            )
        else:
            cmd_parser.add_argument(
                f"--{key}",
                type=type(value),
                # Don't set defaults here - we'll handle priority manually
                default=argparse.SUPPRESS,
            )

    # Also add the use_config argument
    cmd_parser.add_argument(
        "--use_config",
        type=str,
        default=argparse.SUPPRESS,
        help="Path to user config file",
    )

    # Step 6: Parse command line args (only what was explicitly provided)
    cmd_args = vars(cmd_parser.parse_args())

    # Step 7: Create final config with correct priority:
    # Command line args > User config > Default config
    # merged_config already contains default+user config
    final_config = merged_config.copy()

    # Store the original output_base_dir before applying command line args
    original_output_base_dir = final_config["output_base_dir"]

    # Override with any command line arguments
    final_config.update(cmd_args)

    # If output_base_dir was changed via command line, update all dependent paths
    if (
        "output_base_dir" in cmd_args
        and final_config["output_base_dir"] != original_output_base_dir
    ):
        for k, v in final_config.items():
            if isinstance(v, str) and original_output_base_dir in v:
                final_config[k] = v.replace(
                    original_output_base_dir, final_config["output_base_dir"]
                )

    return final_config


def main():
    config = parse_args()

    if config["erase"]:
        print("Erasing previous training outputs")
        shutil.rmtree(config["output_base_dir"], ignore_errors=True)

    if config["verbose"]:
        pprint.pprint(config)
        input("Press Enter to continue...")
    if config["model"] in ["DQN"]:
        setup_and_train_DQN(config)
    elif config["model"] in ["PPO"]:
        setup_and_train_PPO(config)
    elif config["model"] == "explore":
        memory_collector(config)
    elif config["model"] == "evaluate":
        evaluate_reward_system(config)
    else:
        raise ValueError(f"Model {config['model']} not recognized")


if __name__ == "__main__":
    main()
