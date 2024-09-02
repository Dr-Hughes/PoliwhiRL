# -*- coding: utf-8 -*-
import io
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


class PyBoyEnvironment(gym.Env):
    def __init__(self, config, force_window=False):
        super().__init__()
        self.config = config
        self.temp_dir = tempfile.mkdtemp()
        self._fitness = 0
        self.steps = 0
        self.done = False
        self.episode = (
            -2
        )  # because we restart in the constructor, and when the first episode starts
        self.button = 0
        self.actions = ["", "a", "b", "left", "right", "up", "down", "start", "select"]
        self.ignored_buttons = config["ignored_buttons"]
        self.action_space = spaces.Discrete(len(self.actions))
        self.render = config["vision"]
        self.current_max_steps = config["episode_length"]
        files_to_copy = [config["rom_path"], config["state_path"]]
        files_to_copy.extend(
            [file for file in config["extra_files"] if os.path.isfile(file)]
        )

        self.check_files_exist(files_to_copy)

        self.paths = [shutil.copy(file, self.temp_dir) for file in files_to_copy]
        self.state_path = self.paths[1]

        with open(self.state_path, "rb") as state_file:
            state_content = state_file.read()
        self.state_bytes_content = state_content

        self.pyboy = PyBoy(self.paths[0], window="null" if not force_window else "SDL2")
        self.pyboy.rtc_lock_experimental(True) 
        self.pyboy.set_emulation_speed(0)
        self.ram = RAM.RAMManagement(self.pyboy)
        self.reset()

    def get_state_bytes(self):
        return io.BytesIO(self.state_bytes_content)

    def check_files_exist(self, files):
        for file in files:
            if not os.path.isfile(file):
                raise FileNotFoundError(f"File {file} not found.")

    def enable_render(self):
        self.render = True

    def _handle_action(self, action):
        self.button = self.actions[action]
        if self.button not in self.ignored_buttons:
            self.pyboy.button(self.button, delay=15)
        self.pyboy.tick(75, self.render)
        self.steps += 1

    def step(self, action):
        self._handle_action(action)
        self._calculate_fitness()
        observation = (
            self.get_game_area()
            if not self.config["vision"]
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
        self.done = False
        self.pyboy.load_state(self.get_state_bytes())
        self.reward_calculator = Rewards(
            location_goals=self.config["location_goals"],
            pokedex_goals=self.config["pokedex_goals"],
            N_goals_target=self.config["N_goals_target"],
            max_steps=self.config["episode_length"],
            break_on_goal=self.config["break_on_goal"],
        )
        self._fitness = 0
        self._handle_action(0)
        observation = (
            self.get_game_area()
            if not self.config["vision"]
            else self.get_screen_image()
        )
        self.steps = 0
        self.episode += 1
        self.render = self.config["vision"]
        self._calculate_fitness()
        return observation

    def close(self):
        self.pyboy.stop()

    def get_RAM_variables(self):
        return self.ram.get_variables()

    def get_screen_image(self, no_resize=False):
        pil_image = self.pyboy.screen.image

        # Convert PIL image to numpy array (this will be in HWC format)
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

    def save_step_img_data(self, fldr, outdir="./Training Outputs/Runs"):
        record_step(
            self.episode,
            self.steps,
            self.pyboy.screen.image,
            self.button,
            self._fitness,
            fldr,
            outdir,
        )

    def save_state(self, save_path, save_name):
        # create folder if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path + "/" + save_name, "wb") as stateFile:
            self.pyboy.save_state(stateFile)

    def save_gym_state(self, save_path):
        # Create folder if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save PyBoy emulator state to a bytes buffer
        emulator_state_buffer = io.BytesIO()
        self.pyboy.save_state(emulator_state_buffer)
        emulator_state_bytes = emulator_state_buffer.getvalue()

        # Prepare gym environment state
        gym_state = {
            "steps": self.steps,
            "episode": self.episode,
            "button": self.button,
            "_fitness": self._fitness,
            "done": self.done,
            "render": self.render,
            "reward_calculator": self.reward_calculator,
        }

        # Save combined state
        with open(save_path, "wb") as f:
            pickle.dump(
                {"emulator_state": emulator_state_bytes, "gym_state": gym_state}, f
            )

    def load_gym_state(self, load_path):
        with open(load_path, "rb") as f:
            combined_state = pickle.load(f)

        # Load PyBoy emulator state
        self.state_bytes = io.BytesIO(combined_state["emulator_state"])
        self.pyboy.load_state(self.state_bytes)
        self.state_bytes_content = self.state_bytes.getvalue()

        # Load gym environment state
        gym_state = combined_state["gym_state"]
        self.steps = gym_state["steps"]
        self.episode = gym_state["episode"]
        self.button = gym_state["button"]
        self._fitness = gym_state["_fitness"]
        self.done = gym_state["done"]
        self.render = gym_state["render"]
        self.reward_calculator = gym_state["reward_calculator"]

        self.step(0) # Take a step to update things properly
        # Return the loaded state for verification if needed
        return combined_state
