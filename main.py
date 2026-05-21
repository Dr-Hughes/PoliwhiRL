# -*- coding: utf-8 -*-

from PoliwhiRL import setup_and_train_PPO
from PoliwhiRL.explorer import memory_collector
from PoliwhiRL.reward_evaluation import evaluate_reward_system
from PoliwhiRL.evaluator import run_inference
import os
import shutil
import argparse
import json
import pprint
from glob import glob


class StoreBooleanAction(argparse.Action):
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


def load_user_config(config_path, _seen=None):
    if not (config_path and os.path.exists(config_path)):
        return {}

    abs_path = os.path.abspath(config_path)
    if _seen is None:
        _seen = set()
    if abs_path in _seen:
        chain = " -> ".join(list(_seen) + [abs_path])
        raise ValueError(f"Circular 'extends' chain in user config: {chain}")
    _seen.add(abs_path)

    with open(abs_path, "r") as f:
        config = json.load(f)

    parent_ref = config.pop("extends", None)
    if parent_ref is None:
        return config

    if not os.path.isabs(parent_ref):
        parent_ref = os.path.normpath(
            os.path.join(os.path.dirname(abs_path), parent_ref)
        )
    if not os.path.exists(parent_ref):
        raise FileNotFoundError(
            f"Config '{abs_path}' extends '{parent_ref}', which does not exist"
        )

    parent_config = load_user_config(parent_ref, _seen=_seen)
    parent_config.update(config)
    return parent_config


def merge_configs(default_config, user_config):
    merged_config = default_config.copy()
    merged_config.update(user_config)

    old_base = default_config["output_base_dir"]
    new_base = merged_config["output_base_dir"]

    if old_base != new_base:
        # Only remap keys whose default value was output-base-relative. This
        # avoids (a) double-rewriting output_base_dir itself, and (b) clobbering
        # input paths like load_checkpoint that the user pointed at a different
        # run's output dir.
        for k, default_v in default_config.items():
            if k == "output_base_dir":
                continue
            if not (isinstance(default_v, str) and default_v.startswith(old_base)):
                continue
            v = merged_config[k]
            if isinstance(v, str) and old_base in v:
                merged_config[k] = v.replace(old_base, new_base, 1)

    return merged_config


def parse_args():
    default_config = load_default_config()

    initial_parser = argparse.ArgumentParser()
    initial_parser.add_argument(
        "--use_config", type=str, default=None, help="Path to user config file"
    )
    initial_args, _ = initial_parser.parse_known_args()

    user_config = load_user_config(initial_args.use_config)
    merged_config = merge_configs(default_config, user_config)

    cmd_parser = argparse.ArgumentParser()

    for key, value in merged_config.items():
        if isinstance(value, bool):
            cmd_parser.add_argument(
                f"--{key}",
                type=str,
                action=StoreBooleanAction,
                default=argparse.SUPPRESS,
            )
        else:
            cmd_parser.add_argument(
                f"--{key}",
                type=type(value),
                default=argparse.SUPPRESS,
            )

    cmd_parser.add_argument(
        "--use_config",
        type=str,
        default=argparse.SUPPRESS,
        help="Path to user config file",
    )

    cmd_args = vars(cmd_parser.parse_args())
    final_config = merged_config.copy()

    original_output_base_dir = final_config["output_base_dir"]
    final_config.update(cmd_args)

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

    if config["model"] == "PPO":
        setup_and_train_PPO(config)
    elif config["model"] == "explore":
        memory_collector(config)
    elif config["model"] == "reward_eval":
        evaluate_reward_system(config)
    elif config["model"] == "inference":
        run_inference(config)
    else:
        raise ValueError(f"Model {config['model']} not recognized")


if __name__ == "__main__":
    main()
