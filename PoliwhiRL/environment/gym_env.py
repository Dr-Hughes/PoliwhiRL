# -*- coding: utf-8 -*-
import io
import os
import shutil
import pickle
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
from . import RAM
from PoliwhiRL.utils.visuals import record_step
from .rewards import Rewards
from pyboy import PyBoy
from PoliwhiRL.utils.resource_manager import get_resource_pool, create_config_hash


class PyBoyEnvironment(gym.Env):
    def __init__(self, config, force_window=False, shared_temp_dir=None):
        super().__init__()
        self.config = config
        self.config_hash = create_config_hash(config)
        self.resource_pool = get_resource_pool()
        self.shared_temp_dir = shared_temp_dir
        self.use_shared_temp = shared_temp_dir is not None
        self._is_closed = False

        if self.use_shared_temp:
            # Use provided shared temp directory
            self.temp_dir = shared_temp_dir
            self.owns_temp_dir = False
        else:
            # Get shared temp directory from resource pool
            self.temp_dir = self.resource_pool.get_shared_temp_dir(self.config_hash)
            self.owns_temp_dir = False  # Resource pool owns it

        # Create process-specific subdirectory
        process_id = os.getpid()
        self.process_temp_dir = os.path.join(self.temp_dir, f"process_{process_id}")
        os.makedirs(self.process_temp_dir, exist_ok=True)

        self.frames_per_action = 90  # needs enough time to pass door transition
        self.button_hold_frames = 15
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
        self.record = False
        self.use_episode_number = True
        self.record_folder = None
        self.current_max_steps = config["episode_length"]

        files_to_copy = [config["rom_path"], config["state_path"]]
        files_to_copy.extend(
            [file for file in config["extra_files"] if os.path.isfile(file)]
        )

        self.check_files_exist(files_to_copy)

        # Copy files to process-specific temp directory
        self.paths = []
        for file in files_to_copy:
            dest_path = os.path.join(self.process_temp_dir, os.path.basename(file))
            # Always copy to ensure each process has its own files
            shutil.copy(file, dest_path)
            self.paths.append(dest_path)

        self.state_path = self.paths[1]

        with open(self.state_path, "rb") as state_file:
            state_content = state_file.read()
        self.state_bytes_content = state_content

        self.pyboy = PyBoy(
            self.paths[0],
            window="null" if not force_window else "SDL2",
            sound_emulated=False,
        )
        self.pyboy.rtc_lock_experimental(True)
        self.pyboy.set_emulation_speed(0)
        self.ram = RAM.RAMManagement(self.pyboy)
        self.reset()

    @classmethod
    def create_with_shared_temp(cls, config, force_window=False):
        """Factory method to create environment with shared temporary directory"""
        resource_pool = get_resource_pool()
        config_hash = create_config_hash(config)
        shared_temp_dir = resource_pool.get_shared_temp_dir(config_hash)
        return cls(config, force_window, shared_temp_dir)

    def get_state_bytes(self):
        return io.BytesIO(self.state_bytes_content)

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
        if not self.config["vision"]:
            return self.get_game_area().shape
        return self.get_screen_image().shape

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
        return (
            self.get_game_area()
            if not self.config["vision"]
            else self.get_screen_image()
        )

    def reset(self):
        self.button = 0
        self.done = False
        self.record = False
        self.record_folder = None
        self.pyboy.load_state(self.get_state_bytes())
        self.reward_calculator = Rewards(self.config)
        self._fitness = 0
        self._handle_action(0)
        observation = self.get_observation()
        self.steps = 0
        self.episode += 1
        self.render = self.config["vision"]
        self._calculate_fitness()
        return observation

    def close(self):
        if self._is_closed:
            return

        self._is_closed = True

        # Stop the PyBoy emulator
        try:
            if hasattr(self, "pyboy"):
                self.pyboy.stop()
                # Give PyBoy time to save its .ram file before cleaning up temp directory
                import time
                time.sleep(0.1)  # Small delay to ensure PyBoy finishes cleanup
        except Exception as e:
            print(f"Error stopping PyBoy: {e}")

        # Clean up process-specific temp directory
        try:
            if hasattr(self, "process_temp_dir") and os.path.exists(
                self.process_temp_dir
            ):
                # Only delete non-critical files, let .ram files be cleaned up naturally
                for file in os.listdir(self.process_temp_dir):
                    file_path = os.path.join(self.process_temp_dir, file)
                    try:
                        if not file.endswith('.ram'):  # Don't force-delete .ram files
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                    except Exception:
                        pass  # Ignore individual file cleanup errors
                
                # Try to remove the directory, but don't force it if .ram files are still there
                try:
                    os.rmdir(self.process_temp_dir)
                except OSError:
                    pass  # Directory not empty or still in use
        except Exception as e:
            print(f"Error cleaning up process temp directory: {e}")

        # Release reference to shared temp directory
        if not self.owns_temp_dir and hasattr(self, "config_hash"):
            self.resource_pool.release_temp_dir(self.config_hash)

    def __getstate__(self):
        """Custom pickle serialization to handle non-pickleable objects"""
        state = self.__dict__.copy()
        # Remove non-pickleable objects
        state.pop("resource_pool", None)
        state.pop("pyboy", None)
        state.pop("ram", None)
        state.pop("reward_calculator", None)
        return state

    def __setstate__(self, state):
        """Custom pickle deserialization"""
        self.__dict__.update(state)
        # Reinitialize non-pickleable objects
        self.resource_pool = get_resource_pool()
        self._is_closed = False

        # Recreate PyBoy instance
        self.pyboy = PyBoy(
            self.paths[0],
            window="null",
            sound_emulated=False,
        )
        self.pyboy.rtc_lock_experimental(True)
        self.pyboy.set_emulation_speed(0)
        self.ram = RAM.RAMManagement(self.pyboy)
        self.reward_calculator = Rewards(self.config)

    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            if not self._is_closed:
                self.close()
        except Exception:
            pass

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

    def get_location_data(self):
        """Get current location data for exploration memory bank"""
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

    def load_gym_state(self, load_path, updated_steps=None, updated_n_goals=None):
        try:
            with open(load_path, "rb") as f:
                combined_state = pickle.load(f)
        except FileNotFoundError:
            print("Could not find file at path:", load_path)
            print("Returning to initial state.")
            return self.reset()

        # Create a new BytesIO object and store its content
        emulator_state_bytes = combined_state["emulator_state"]
        state_bytes_io = io.BytesIO(emulator_state_bytes)

        # Load the state
        self.pyboy.load_state(state_bytes_io)

        # Update the content for future resets
        self.state_bytes_content = emulator_state_bytes

        # Load gym environment state
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
