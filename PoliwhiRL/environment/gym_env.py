# -*- coding: utf-8 -*-
import io
import math
import os
import pickle
import shutil
import tempfile
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
from . import RAM
from PoliwhiRL.utils.visuals import record_step
from .rewards import Rewards
from pyboy import PyBoy


# Stable ordering for the RAM observation vector. Treat this as a contract:
# changing the order or removing an entry will invalidate trained models.
# New features should be appended to the end. The model's RAM encoder reads
# its input dim from RAM_OBS_DIM at startup, so additions here are
# automatically picked up by the model.
STORY_FLAGS_NUM_BYTES = 256  # RAM region 0xDA72–0xDB71

_BASE_RAM_FEATURE_KEYS = (
    "x",
    "y",
    "map_num",
    "map_bank",
    "room",
    "warp_number",
    "party_level",
    "party_hp",
    "party_exp",
    "money",
    "pokedex_seen",
    "pokedex_owned",
    "collision_down",
    "collision_up",
    "collision_left",
    "collision_right",
    # Goal-conditioning: target_x, target_y, target_map come from the active
    # Rewards goal; has_active_target is 1 while a location goal is pending.
    "target_x",
    "target_y",
    "target_map",
    "has_active_target",
    # Per-episode exploration summary so the policy has at least a rough
    # signal for "have I been finding new tiles."
    "explored_tile_count",
)
# Story-flag bytes are appended after the base features. Each byte is one
# of 256 entries in the 0xDA72–0xDB71 region; each holds 8 individual story
# flags as a bitfield. We expose them as bytes (one float per byte, /255)
# rather than expanding to 2048 bits — the policy learns relevant patterns
# from byte values.
RAM_FEATURE_KEYS = _BASE_RAM_FEATURE_KEYS + tuple(
    f"story_flag_byte_{i:03d}" for i in range(STORY_FLAGS_NUM_BYTES)
)
RAM_OBS_DIM = len(RAM_FEATURE_KEYS)
_BASE_RAM_LEN = len(_BASE_RAM_FEATURE_KEYS)


def _build_ram_vector(env_vars, target, explored_tile_count):
    """Pack RAM + goal-conditioning + exploration scalars into a fixed-order,
    ~[0, 1]-scaled float32 vector. Single source of truth — env, tests, any
    eval tool should construct the vector via this function.

    Parameters
    ----------
    env_vars : dict
        Output of RAM.RAMManagement.get_variables().
    target : tuple
        (target_x, target_y, target_map, has_active_target) from
        Rewards.get_current_target_vector().
    explored_tile_count : int
        len(Rewards.explored_tiles) at this step.
    """
    party_level, party_hp, party_exp = env_vars["party_info"]
    target_x, target_y, target_map, has_active = target
    base = np.array(
        [
            env_vars["X"] / 255.0,
            env_vars["Y"] / 255.0,
            env_vars["map_num"] / 255.0,
            env_vars["map_bank"] / 255.0,
            env_vars["room"] / 255.0,
            env_vars["warp_number"] / 255.0,
            party_level / 100.0,
            party_hp / 1000.0,
            math.log1p(max(0, party_exp)) / 20.0,
            env_vars["money"] / 1_000_000.0,
            env_vars["pokedex_seen"] / 251.0,
            env_vars["pokedex_owned"] / 251.0,
            env_vars["collision_down"] / 255.0,
            env_vars["collision_up"] / 255.0,
            env_vars["collision_left"] / 255.0,
            env_vars["collision_right"] / 255.0,
            target_x / 255.0,
            target_y / 255.0,
            target_map / 255.0,
            float(has_active),
            # Soft saturation for unbounded count; agent only needs relative
            # sense of "low vs high."
            math.log1p(max(0, explored_tile_count)) / 6.0,
        ],
        dtype=np.float32,
    )
    # Story flags: 256 raw bytes normalised to [0, 1]. Each byte holds 8
    # individual flag bits; the model learns the patterns directly.
    story_flags = np.asarray(env_vars["story_flags"], dtype=np.float32) / 255.0
    if story_flags.size != STORY_FLAGS_NUM_BYTES:
        raise ValueError(
            f"Expected {STORY_FLAGS_NUM_BYTES} story flag bytes, got {story_flags.size}"
        )
    return np.concatenate([base, story_flags])


class PyBoyEnvironment(gym.Env):
    def __init__(self, config, force_window=False):
        super().__init__()
        self.config = config
        self._is_closed = False

        self.frames_per_action = 90
        self.button_hold_frames = 15
        self._fitness = 0
        self.steps = 0
        self.done = False
        self.episode = -2
        self.button = 0
        self.actions = ["", "a", "b", "left", "right", "up", "down", "start", "select"]
        self.ignored_buttons = config["ignored_buttons"]
        self.action_space = spaces.Discrete(len(self.actions))
        self.render = config["vision"]
        self.record = False
        self.use_episode_number = True
        self.record_folder = None
        self.current_max_steps = config["episode_length"]

        files_to_copy = [config["rom_path"], config["state_path"]]
        files_to_copy.extend(
            [
                file
                for file in config["extra_files"]
                if os.path.isfile(file) and os.path.getsize(file) > 0
            ]
        )

        self.check_files_exist(files_to_copy)

        self.paths = list(files_to_copy)
        self.state_path = self.paths[1]

        with open(self.state_path, "rb") as state_file:
            state_content = state_file.read()
        self.state_bytes_content = state_content

        # Copy ROM (and any sidecars) into a per-instance temp dir so PyBoy's
        # .ram/.rtc writes don't mutate the canonical files in emu_files/.
        self._tmpdir = tempfile.TemporaryDirectory(prefix="poliwhirl_emu_")
        rom_dst = os.path.join(self._tmpdir.name, os.path.basename(self.paths[0]))
        shutil.copy(self.paths[0], rom_dst)
        for extra in files_to_copy[2:]:
            shutil.copy(extra, os.path.join(self._tmpdir.name, os.path.basename(extra)))
        self.paths[0] = rom_dst

        try:
            self.pyboy = PyBoy(
                self.paths[0],
                window="null" if not force_window else "SDL2",
                sound_emulated=False,
            )
        except Exception:
            self._tmpdir.cleanup()
            raise
        self.pyboy.rtc_lock_experimental(True)
        self.pyboy.set_emulation_speed(0)
        self.ram = RAM.RAMManagement(self.pyboy)
        self.reset()

    def get_state_bytes(self):
        return io.BytesIO(self.state_bytes_content)

    def set_state_path(self, path):
        """Swap in a new save-state. Takes effect on the next reset() so
        the current episode finishes normally. Used by VecPyBoyEnv for
        per-episode state cycling across a pool of save-states.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"State file not found: {path}")
        with open(path, "rb") as f:
            self.state_bytes_content = f.read()
        self.state_path = path

    def replay_actions(self, actions):
        """Walk the env forward by replaying a sequence of actions.

        Used immediately after env.reset() to warm-start the env state +
        Rewards object to a curriculum-aligned position. The actions are
        executed without storing transitions; rewards accrued during replay
        are *not* counted toward the training episode.

        After replay:
            - PyBoy memory is at the post-replay position.
            - Rewards.current_goal_index, N_goals, pokedex_* reflect what
              was achieved during replay (so the next goal target is the
              one *after* the replay's reach).
            - Per-episode fields (steps, explored_tiles, cumulative_reward)
              are cleared via Rewards.start_new_episode() so the training
              episode that follows has a clean step budget.
            - env.steps and env._fitness are reset to 0.
        """
        if not actions:
            return self.get_observation()
        for a in actions:
            self._handle_action(int(a))
            self._calculate_fitness()
        # Promote curriculum state forward; clear per-episode counters.
        self.reward_calculator.start_new_episode()
        self.steps = 0
        self._fitness = 0
        self.done = False
        return self.get_observation()

    def check_files_exist(self, files):
        for file in files:
            if not os.path.isfile(file):
                raise FileNotFoundError(f"File {file} not found.")

    def enable_render(self):
        self.render = True

    def _handle_action(self, action):
        frames = self.frames_per_action
        self.button = self.actions[action]
        if self.button not in self.ignored_buttons:
            self.pyboy.button(self.button, delay=self.button_hold_frames)
            frames -= self.button_hold_frames
        self.pyboy.tick(frames, self.render)
        self.steps += 1

    def step(self, action):
        self._handle_action(action)
        self._calculate_fitness()
        observation = self.get_observation()

        if self.record:
            self.save_step_img_data(
                self.record_folder, outdir=self.config["record_path"]
            )

        return observation, self._fitness, self.done, False

    def output_shape(self):
        """Image observation shape (C, H, W)."""
        if not self.config["vision"]:
            return self.get_game_area().shape
        return self.get_screen_image().shape

    def ram_observation_shape(self):
        return (RAM_OBS_DIM,)

    def get_game_area(self):
        return self.pyboy.game_area()[:18, :20].astype(np.uint8)

    def get_screen_size(self):
        return self.get_screen_image().shape

    def enable_record(self, folder, use_episode_number=True):
        self.use_episode_number = use_episode_number
        self.record = True
        self.record_folder = folder
        self.enable_render()

    def _calculate_fitness(self):
        self._fitness, reward_done = self.reward_calculator.calculate_reward(
            self.ram.get_variables(), self.button
        )
        if reward_done:
            self.done = True

    def get_observation(self):
        """Multi-modal observation dict {"image": ndarray, "ram": ndarray}.

        The image preserves the original screen / tilemap output for the
        CNN. The RAM vector packs position, party state, goal target, and
        exploration summary for the policy to condition on directly.
        """
        image = (
            self.get_game_area()
            if not self.config["vision"]
            else self.get_screen_image()
        )
        ram = _build_ram_vector(
            self.ram.get_variables(),
            self.reward_calculator.get_current_target_vector(),
            self.reward_calculator.explored_tile_count(),
        )
        return {"image": image, "ram": ram}

    def reset(self):
        self.button = 0
        self.done = False
        self.record = False
        self.record_folder = None
        self.pyboy.load_state(self.get_state_bytes())
        self.reward_calculator = Rewards(self.config)
        self._fitness = 0
        self._handle_action(0)
        self.steps = 0
        self.episode += 1
        self.render = self.config["vision"]
        # Compute fitness BEFORE building the observation so the goal-target
        # field in the RAM vector reflects any goal advance that fired on the
        # no-op startup step.
        self._calculate_fitness()
        return self.get_observation()

    def close(self):
        if self._is_closed:
            return
        self._is_closed = True

        try:
            if hasattr(self, "pyboy"):
                self.pyboy.stop()
        except Exception as e:
            print(f"Error stopping PyBoy: {e}")

        try:
            if hasattr(self, "_tmpdir"):
                self._tmpdir.cleanup()
        except Exception as e:
            print(f"Error cleaning emu temp dir: {e}")

    def get_screen_image(self, no_resize=False):
        pil_image = self.pyboy.screen.image
        numpy_image = np.array(pil_image)[:, :, :3]

        use_grayscale = self.config["use_grayscale"]
        scaling_factor = self.config["scaling_factor"]

        if scaling_factor != 1.0 and not no_resize:
            new_width = int(numpy_image.shape[1] * scaling_factor)
            new_height = int(numpy_image.shape[0] * scaling_factor)
            numpy_image = cv2.resize(
                numpy_image, (new_width, new_height), interpolation=cv2.INTER_AREA
            )

        if use_grayscale:
            numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2GRAY)
            numpy_image = np.expand_dims(numpy_image, axis=-1)

        if use_grayscale:
            numpy_image = numpy_image.transpose(2, 0, 1)
        else:
            numpy_image = numpy_image.transpose(2, 0, 1)

        return numpy_image.astype(np.uint8)

    def get_pyboy_bg(self):
        return np.array(self.pyboy.tilemap_background[:18, :20])

    def get_pyboy_wnd(self):
        return np.array(self.pyboy.tilemap_window[:18, :20])

    def get_location_data(self):
        variables = self.ram.get_variables()
        return {
            "x": variables["X"],
            "y": variables["Y"],
            "map_num": variables["map_num"],
            "room": variables["room"],
        }

    def save_step_img_data(self, fldr, outdir="./Training Outputs/Runs"):
        record_step(
            self.episode if self.use_episode_number else -1,
            self.steps,
            self.pyboy.screen.image,
            self.button,
            self._fitness,
            fldr,
            outdir,
        )

    def save_state(self, save_path, save_name):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path + "/" + save_name, "wb") as stateFile:
            self.pyboy.save_state(stateFile)

    def save_gym_state(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        emulator_state_buffer = io.BytesIO()
        self.pyboy.save_state(emulator_state_buffer)
        emulator_state_bytes = emulator_state_buffer.getvalue()

        gym_state = {
            "steps": self.steps,
            "episode": self.episode,
            "button": self.button,
            "_fitness": self._fitness,
            "done": self.done,
            "render": self.render,
            "reward_calculator": self.reward_calculator,
        }

        with open(save_path, "wb") as f:
            pickle.dump(
                {"emulator_state": emulator_state_bytes, "gym_state": gym_state}, f
            )

    def load_gym_state(self, load_path, updated_steps=None, updated_n_goals=None):
        try:
            with open(load_path, "rb") as f:
                combined_state = pickle.load(f)
        except FileNotFoundError:
            print("Could not find file at path:", load_path)
            print("Returning to initial state.")
            return self.reset()

        emulator_state_bytes = combined_state["emulator_state"]
        state_bytes_io = io.BytesIO(emulator_state_bytes)
        self.pyboy.load_state(state_bytes_io)
        self.state_bytes_content = emulator_state_bytes

        gym_state = combined_state["gym_state"]
        self.steps = gym_state["steps"]
        self.episode = gym_state["episode"]
        self.button = gym_state["button"]
        self._fitness = gym_state["_fitness"]
        self.done = gym_state["done"]
        self.render = gym_state["render"]
        self.reward_calculator = gym_state["reward_calculator"]

        if updated_steps and updated_n_goals:
            self.reward_calculator.update_targets(updated_n_goals, updated_steps)
            self.done = False

        return self.get_observation()
