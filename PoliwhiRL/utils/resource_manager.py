# -*- coding: utf-8 -*-
"""Resource management utilities for PPO multi-agent training"""
import os
import tempfile
import shutil
import threading
import fcntl
import time
import hashlib
import atexit
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict
import psutil


class ResourcePool:
    """Manages pooled resources (temp directories, file handles) for multi-agent training"""

    def __init__(self):
        self._temp_dirs = {}
        self._temp_dir_lock = threading.Lock()
        self._file_locks = defaultdict(threading.Lock)
        self._cleanup_registered = False
        self._register_cleanup()

    def _register_cleanup(self):
        """Register cleanup handlers"""
        if not self._cleanup_registered:
            atexit.register(self.cleanup_all)
            self._cleanup_registered = True

    def get_shared_temp_dir(self, config_hash):
        """Get or create a shared temporary directory"""
        with self._temp_dir_lock:
            if config_hash not in self._temp_dirs:
                temp_dir = tempfile.mkdtemp(prefix=f"poliwhirl_{config_hash}_")
                self._temp_dirs[config_hash] = {
                    "path": temp_dir,
                    "ref_count": 0,
                    "last_access": time.time(),
                }

            self._temp_dirs[config_hash]["ref_count"] += 1
            self._temp_dirs[config_hash]["last_access"] = time.time()
            return self._temp_dirs[config_hash]["path"]

    def release_temp_dir(self, config_hash):
        """Release a reference to a temporary directory"""
        with self._temp_dir_lock:
            if config_hash in self._temp_dirs:
                self._temp_dirs[config_hash]["ref_count"] -= 1
                if self._temp_dirs[config_hash]["ref_count"] <= 0:
                    # Schedule for cleanup after a delay
                    self._temp_dirs[config_hash]["cleanup_time"] = (
                        time.time() + 60
                    )  # 1 minute delay

    def cleanup_unused_dirs(self):
        """Clean up temporary directories that are no longer in use"""
        with self._temp_dir_lock:
            current_time = time.time()
            to_remove = []

            for config_hash, info in self._temp_dirs.items():
                if (
                    info["ref_count"] <= 0
                    and "cleanup_time" in info
                    and current_time >= info["cleanup_time"]
                ):
                    try:
                        if os.path.exists(info["path"]):
                            shutil.rmtree(info["path"])
                        to_remove.append(config_hash)
                    except Exception as e:
                        print(f"Error cleaning up temp dir: {e}")

            for config_hash in to_remove:
                del self._temp_dirs[config_hash]

    def cleanup_all(self):
        """Clean up all temporary directories"""
        with self._temp_dir_lock:
            for info in self._temp_dirs.values():
                try:
                    if os.path.exists(info["path"]):
                        shutil.rmtree(info["path"])
                except Exception as e:
                    print(f"Error during final cleanup: {e}")
            self._temp_dirs.clear()

    @contextmanager
    def file_lock(self, filepath):
        """Context manager for file locking"""
        lock_file = f"{filepath}.lock"
        fd = None
        try:
            # Create lock file if it doesn't exist
            Path(lock_file).touch()
            fd = os.open(lock_file, os.O_RDWR | os.O_CREAT)

            # Acquire exclusive lock
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            # Release lock
            if fd is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
                try:
                    os.remove(lock_file)
                except OSError:
                    pass


# Global resource pool instance
_resource_pool = ResourcePool()


def get_resource_pool():
    """Get the global resource pool instance"""
    return _resource_pool


class ProcessMonitor:
    """Monitors agent processes and tracks their progress"""

    def __init__(self):
        self.process_info = {}
        self.lock = threading.Lock()

    def register_process(self, agent_id, process_id):
        """Register a new agent process"""
        with self.lock:
            self.process_info[agent_id] = {
                "pid": process_id,
                "start_time": time.time(),
                "last_heartbeat": time.time(),
                "episodes_completed": 0,
                "is_alive": True,
                "memory_usage": 0,
                "cpu_percent": 0,
            }

    def update_heartbeat(self, agent_id, episodes_completed=None):
        """Update process heartbeat"""
        with self.lock:
            if agent_id in self.process_info:
                self.process_info[agent_id]["last_heartbeat"] = time.time()
                if episodes_completed is not None:
                    self.process_info[agent_id][
                        "episodes_completed"
                    ] = episodes_completed

    def update_resource_usage(self, agent_id):
        """Update process resource usage"""
        with self.lock:
            if agent_id in self.process_info:
                try:
                    process = psutil.Process(self.process_info[agent_id]["pid"])
                    self.process_info[agent_id][
                        "memory_usage"
                    ] = process.memory_info().rss
                    # Get CPU percent with interval for accurate reading
                    self.process_info[agent_id]["cpu_percent"] = process.cpu_percent(
                        interval=0.1
                    )
                except psutil.NoSuchProcess:
                    self.process_info[agent_id]["is_alive"] = False

    def get_timeout_for_agent(self, agent_id, base_timeout=180):
        """Calculate dynamic timeout based on agent progress"""
        with self.lock:
            if agent_id not in self.process_info:
                return base_timeout

            info = self.process_info[agent_id]

            # If agent has been making progress, give it more time
            if info["episodes_completed"] > 0:
                # Add 30 seconds per episode completed
                dynamic_timeout = base_timeout + (info["episodes_completed"] * 30)
                # Cap at 10 minutes
                return min(dynamic_timeout, 600)

            # If agent just started, give it the base timeout
            time_since_start = time.time() - info["start_time"]
            if time_since_start < base_timeout:
                return base_timeout

            # Otherwise, use base timeout
            return base_timeout

    def check_agent_health(self, agent_id):
        """Check if an agent is healthy"""
        with self.lock:
            if agent_id not in self.process_info:
                # Agent not registered yet, assume healthy
                return True

            info = self.process_info[agent_id]

            # Check if process is alive
            if not info["is_alive"]:
                return False

            # Check memory usage (terminate if > 4GB)
            if info["memory_usage"] > 4 * 1024 * 1024 * 1024:
                print(
                    f"Agent {agent_id} using too much memory: {info['memory_usage'] / 1024 / 1024 / 1024:.2f}GB"
                )
                return False

            return True

    def get_all_agents_status(self):
        """Get status summary of all agents"""
        with self.lock:
            return {
                agent_id: {
                    "episodes": info["episodes_completed"],
                    "uptime": time.time() - info["start_time"],
                    "memory_mb": info["memory_usage"] / 1024 / 1024,
                    "cpu_percent": info["cpu_percent"],
                    "is_alive": info["is_alive"],
                }
                for agent_id, info in self.process_info.items()
            }


class SharedMemoryManager:
    """Manages shared memory for model parameters"""

    def __init__(self):
        self.shared_tensors = {}
        self.lock = threading.Lock()

    def create_shared_tensor(self, name, tensor):
        """Create a shared memory tensor"""
        with self.lock:
            # Convert to CPU tensor if needed
            if tensor.is_cuda:
                tensor = tensor.cpu()

            # Create shared memory tensor
            shared_tensor = tensor.share_memory_()
            self.shared_tensors[name] = shared_tensor
            return shared_tensor

    def get_shared_tensor(self, name):
        """Get a shared memory tensor"""
        with self.lock:
            return self.shared_tensors.get(name)

    def update_shared_tensor(self, name, new_tensor):
        """Update a shared tensor with new values"""
        with self.lock:
            if name in self.shared_tensors:
                # Copy data to shared tensor
                self.shared_tensors[name].copy_(new_tensor)

    def cleanup(self):
        """Clean up shared memory"""
        with self.lock:
            self.shared_tensors.clear()


def create_config_hash(config):
    """Create a hash from relevant config parameters"""
    relevant_keys = ["rom_path", "state_path", "extra_files"]
    config_str = str(sorted([(k, config.get(k)) for k in relevant_keys if k in config]))
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def periodic_cleanup_thread(resource_pool, interval=60):
    """Thread function for periodic resource cleanup"""
    while True:
        time.sleep(interval)
        try:
            resource_pool.cleanup_unused_dirs()
        except Exception as e:
            print(f"Error in periodic cleanup: {e}")
