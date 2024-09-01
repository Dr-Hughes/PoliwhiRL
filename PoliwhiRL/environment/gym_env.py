# -*- coding: utf-8 -*-
import os
import shutil
import tempfile
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
from . import RAM
from PoliwhiRL.utils.utils import document
from .rewards import Rewards
from pyboy import PyBoy

actions = ["", "a", "b", "left", "right", "up", "down", "start", "select"]


class PyBoyEnvironment(gym.Env):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.temp_dir = tempfile.mkdtemp()
        self._fitness = 0
        self.steps = 0
        self.episode = -1
        self.button = 0
        self.ignored_buttons = config.get("ignored_buttons", ["", "start", "select"])
        self.action_space = spaces.Discrete(len(actions))
        self.render = config.get("vision", False)
        self.current_max_steps = config.get("episode_length", 100)

        files_to_copy = [config.get("rom_path"), config.get("state_path")]
        files_to_copy.extend(
            [file for file in config.get("extra_files", []) if os.path.isfile(file)]
        )

        self.check_files_exist(files_to_copy)

        self.paths = [shutil.copy(file, self.temp_dir) for file in files_to_copy]
        self.state_path = self.paths[1]

        self.pyboy = PyBoy(self.paths[0], window="null")
        self.pyboy.set_emulation_speed(0)
        self.ram = RAM.RAMManagement(self.pyboy)
        self.pyboy.set_emulation_speed(0)
        self.reset()

    def check_files_exist(self, files):
        for file in files:
            if not os.path.isfile(file):
                raise FileNotFoundError(f"File {file} not found.")

    def enable_render(self):
        self.render = True

    def handle_action(self, action):
        self.button = actions[action]
        if self.button not in self.ignored_buttons:
            self.pyboy.button_press(self.button)
            self.pyboy.tick(15, False)
            self.pyboy.button_release(self.button)
        self.pyboy.tick(75, self.render)
        self.steps += 1
        self.done = self.steps == self.config.get("episode_length", 100)

    def step(self, action):
        self.handle_action(action)
        self._calculate_fitness()
        observation = (
            self.get_game_area()
            if not self.config.get("vision", False)
            else self.get_screen_image()
        )
        return observation, self._fitness, self.done, False

    def get_game_area(self):
        return self.pyboy.game_area()[:18, :20]

    def get_screen_size(self):
        return self.get_screen_image().shape

    def _calculate_fitness(self):
        self._fitness, reward_done = self.reward_calculator.calculate_reward(
            self.get_RAM_variables(), self.button
        )
        if reward_done:
            self.done = True

    def reset(self):
        self.button = 0
        with open(self.state_path, "rb") as stateFile:
            self.pyboy.load_state(stateFile)
        self.reward_calculator = Rewards(
            self.config.get("reward_goals", None),
            self.config.get("N_goals_target", 2),
            self.config.get("episode_length", 100),
            self.config.get("break_on_goal", True),
        )
        self._fitness = 0
        self.handle_action(0)
        observation = (
            self.get_game_area()
            if not self.config.get("vision", False)
            else self.get_screen_image()
        )
        self.steps = 0
        self.episode += 1
        self.render = self.config.get("vision", False)
        self._calculate_fitness()
        return observation

    def close(self):
        self.pyboy.stop()

    def save_state(self, save_path, save_name):
        # create folder if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path + "/" + save_name, "wb") as stateFile:
            self.pyboy.save_state(stateFile)

    def get_RAM_variables(self):
        return self.ram.get_variables()

    def get_screen_image(self, no_resize=False):
        pil_image = self.pyboy.screen.image

        # Convert PIL image to numpy array (this will be in HWC format)
        numpy_image = np.array(pil_image)[:, :, :3]

        use_grayscale = self.config.get("use_grayscale", False)
        scaling_factor = self.config.get("scaling_factor", 1)

        if scaling_factor != 1.0 and not no_resize:
            new_width = int(numpy_image.shape[1] * scaling_factor)
            new_height = int(numpy_image.shape[0] * scaling_factor)
            numpy_image = cv2.resize(
                numpy_image, (new_width, new_height), interpolation=cv2.INTER_AREA
            )

        if use_grayscale:
            numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2GRAY)
            numpy_image = np.expand_dims(numpy_image, axis=-1)  # Add channel dimension

        # Convert to CHW format
        if use_grayscale:
            numpy_image = numpy_image.transpose(2, 0, 1)
        else:
            numpy_image = numpy_image.transpose(2, 0, 1)

        return numpy_image.astype(np.uint8)

    def get_pyboy_bg(self):
        return np.array(self.pyboy.tilemap_background[:18, :20])

    def get_pyboy_wnd(self):
        return np.array(self.pyboy.tilemap_window[:18, :20])

    def record(self, fldr):
        document(
            self.episode,
            self.steps,
            self.pyboy.screen.image,
            self.button,
            self._fitness,
            fldr,
        )
