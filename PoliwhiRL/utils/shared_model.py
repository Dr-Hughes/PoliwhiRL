# -*- coding: utf-8 -*-
"""Shared memory model management for multi-agent training"""
import torch
import torch.multiprocessing as mp
from threading import Lock
import time


class SharedModelManager:
    """Manages shared memory models for multi-agent training"""

    def __init__(self):
        self.shared_models = {}
        self.model_locks = {}
        self.update_counts = {}
        self.lock = Lock()

    def create_shared_model(self, model_id, model_state_dict):
        """Create a shared memory copy of a model"""
        with self.lock:
            if model_id in self.shared_models:
                return

            # Create shared tensors for each parameter
            shared_state_dict = {}
            for name, param in model_state_dict.items():
                if isinstance(param, torch.Tensor):
                    # Move to CPU and share memory
                    shared_param = param.cpu().share_memory_()
                    shared_state_dict[name] = shared_param
                else:
                    shared_state_dict[name] = param

            self.shared_models[model_id] = shared_state_dict
            self.model_locks[model_id] = mp.Lock()
            self.update_counts[model_id] = mp.Value("i", 0)

    def update_shared_model(self, model_id, agent_state_dict, agent_id, num_agents):
        """Update shared model with averaged parameters from an agent"""
        if model_id not in self.shared_models:
            raise ValueError(f"Model {model_id} not found in shared memory")

        with self.model_locks[model_id]:
            # Increment update count
            self.update_counts[model_id].value += 1
            count = self.update_counts[model_id].value

            # If this is the first update, just copy the parameters
            if count == 1:
                for name, param in agent_state_dict.items():
                    if (
                        isinstance(param, torch.Tensor)
                        and name in self.shared_models[model_id]
                    ):
                        self.shared_models[model_id][name].copy_(param.cpu())
            else:
                # Average with existing parameters
                weight = 1.0 / count
                for name, param in agent_state_dict.items():
                    if (
                        isinstance(param, torch.Tensor)
                        and name in self.shared_models[model_id]
                    ):
                        # Weighted average: old_val * (1 - weight) + new_val * weight
                        self.shared_models[model_id][name].mul_(1 - weight).add_(
                            param.cpu(), alpha=weight
                        )

            # If all agents have updated, reset counter for next iteration
            if count >= num_agents:
                self.update_counts[model_id].value = 0

    def get_shared_model(self, model_id):
        """Get the current shared model state"""
        if model_id not in self.shared_models:
            raise ValueError(f"Model {model_id} not found in shared memory")

        with self.model_locks[model_id]:
            # Create a copy to avoid race conditions
            model_copy = {}
            for name, param in self.shared_models[model_id].items():
                if isinstance(param, torch.Tensor):
                    model_copy[name] = param.clone()
                else:
                    model_copy[name] = param
            return model_copy

    def wait_for_all_updates(self, model_id, num_agents, timeout=300):
        """Wait for all agents to update the model"""
        start_time = time.time()
        while True:
            with self.lock:
                if self.update_counts[model_id].value >= num_agents:
                    return True

            if time.time() - start_time > timeout:
                return False

            time.sleep(0.1)

    def cleanup(self):
        """Clean up shared memory"""
        with self.lock:
            self.shared_models.clear()
            self.model_locks.clear()
            self.update_counts.clear()


# Global shared model manager instance
_shared_model_manager = None
_manager_lock = Lock()


def get_shared_model_manager():
    """Get or create the global shared model manager"""
    global _shared_model_manager
    with _manager_lock:
        if _shared_model_manager is None:
            _shared_model_manager = SharedModelManager()
        return _shared_model_manager


def cleanup_shared_model_manager():
    """Clean up the global shared model manager"""
    global _shared_model_manager
    with _manager_lock:
        if _shared_model_manager is not None:
            _shared_model_manager.cleanup()
            _shared_model_manager = None
