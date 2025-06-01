# -*- coding: utf-8 -*-
"""
Shared memory manager for efficient multi-agent parameter sharing
without constant disk I/O operations.
"""

import torch
import torch.multiprocessing as mp
from typing import Dict, List, Optional, Any, Tuple
import threading
import time
import os
from dataclasses import dataclass
import pickle


@dataclass
class ModelSnapshot:
    """Lightweight snapshot of model parameters"""
    params: Dict[str, torch.Tensor]
    metadata: Dict[str, Any]
    timestamp: float
    episode_count: int


class SharedMemoryModelManager:
    """
    Manages model parameters in shared memory for multi-agent training.
    Provides thread-safe access and parameter averaging capabilities.
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.lock = threading.RLock()
        
        # Current model state
        self._current_params: Dict[str, torch.Tensor] = {}
        self._optimizer_state: Dict = {}
        self._scheduler_state: Dict = {}
        self._episode_count: int = 0
        self._last_update: float = 0
        
        # History for debugging/rollback
        self._parameter_history: List[ModelSnapshot] = []
        self._max_history_size = 5
        
        # Agent tracking
        self._active_agents: Dict[int, Dict] = {}
        self._completed_agents: Dict[int, Dict] = {}
        
    def initialize_from_agent(self, agent) -> None:
        """Initialize shared memory from an existing agent"""
        with self.lock:
            # Extract parameters
            self._current_params = {
                'actor_critic': self._extract_params(agent.model.actor_critic),
                'icm': self._extract_params(agent.model.icm.icm)
            }
            
            self._optimizer_state = agent.model.optimizer.state_dict()
            self._scheduler_state = agent.model.scheduler.state_dict()
            self._episode_count = agent.episode
            self._last_update = time.time()
            
            print(f"Shared memory initialized with {len(self._current_params)} parameter groups")
    
    def _extract_params(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """Extract model parameters ensuring they're on correct device"""
        return {
            name: param.data.clone().to(self.device) 
            for name, param in model.named_parameters()
        }
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current model state for agent initialization"""
        with self.lock:
            return {
                'params': {k: {name: tensor.clone() for name, tensor in group.items()} 
                          for k, group in self._current_params.items()},
                'optimizer_state': self._optimizer_state.copy(),
                'scheduler_state': self._scheduler_state.copy(),
                'episode_count': self._episode_count,
                'timestamp': self._last_update
            }
    
    def update_agent_model(self, agent, agent_id: int) -> None:
        """Update agent with current shared parameters"""
        with self.lock:
            if 'actor_critic' in self._current_params:
                for name, param in agent.model.actor_critic.named_parameters():
                    if name in self._current_params['actor_critic']:
                        param.data.copy_(self._current_params['actor_critic'][name])
            
            if 'icm' in self._current_params:
                for name, param in agent.model.icm.icm.named_parameters():
                    if name in self._current_params['icm']:
                        param.data.copy_(self._current_params['icm'][name])
            
            if self._optimizer_state:
                try:
                    agent.model.optimizer.load_state_dict(self._optimizer_state)
                except Exception as e:
                    print(f"Warning: Could not load optimizer state: {e}")
            
            if self._scheduler_state:
                try:
                    agent.model.scheduler.load_state_dict(self._scheduler_state)
                except Exception as e:
                    print(f"Warning: Could not load scheduler state: {e}")
            
            agent.episode = self._episode_count
            
            # Track agent
            self._active_agents[agent_id] = {
                'last_sync': time.time(),
                'episodes': agent.episode
            }
    
    def submit_agent_results(self, agent_id: int, agent) -> None:
        """Submit trained agent results for averaging"""
        with self.lock:
            result = {
                'params': {
                    'actor_critic': self._extract_params(agent.model.actor_critic),
                    'icm': self._extract_params(agent.model.icm.icm)
                },
                'optimizer_state': agent.model.optimizer.state_dict(),
                'scheduler_state': agent.model.scheduler.state_dict(),
                'episode_count': agent.episode,
                'agent_id': agent_id,
                'timestamp': time.time()
            }
            
            self._completed_agents[agent_id] = result
            
            if agent_id in self._active_agents:
                del self._active_agents[agent_id]
    
    def average_completed_agents(self, expected_agents: Optional[int] = None) -> bool:
        """Average parameters from all completed agents and update shared state"""
        with self.lock:
            if not self._completed_agents:
                return False
            
            if expected_agents and len(self._completed_agents) < expected_agents:
                print(f"Warning: Only {len(self._completed_agents)}/{expected_agents} agents completed")
            
            # Store snapshot for history
            if self._current_params:
                snapshot = ModelSnapshot(
                    params=self._current_params.copy(),
                    metadata={'method': 'pre_averaging'},
                    timestamp=time.time(),
                    episode_count=self._episode_count
                )
                self._parameter_history.append(snapshot)
                
                # Limit history size
                if len(self._parameter_history) > self._max_history_size:
                    self._parameter_history.pop(0)
            
            results = list(self._completed_agents.values())
            num_agents = len(results)
            
            # Average actor-critic parameters
            if 'actor_critic' in results[0]['params']:
                averaged_ac = {}
                for param_name in results[0]['params']['actor_critic']:
                    averaged_ac[param_name] = sum(
                        result['params']['actor_critic'][param_name] 
                        for result in results
                    ) / num_agents
                self._current_params['actor_critic'] = averaged_ac
            
            # Average ICM parameters
            if 'icm' in results[0]['params']:
                averaged_icm = {}
                for param_name in results[0]['params']['icm']:
                    averaged_icm[param_name] = sum(
                        result['params']['icm'][param_name] 
                        for result in results
                    ) / num_agents
                self._current_params['icm'] = averaged_icm
            
            # Average optimizer states (numerical values only)
            avg_optimizer = results[0]['optimizer_state'].copy()
            if 'state' in avg_optimizer:
                for key in avg_optimizer['state']:
                    for param_key in avg_optimizer['state'][key]:
                        if isinstance(avg_optimizer['state'][key][param_key], torch.Tensor):
                            avg_optimizer['state'][key][param_key] = sum(
                                result['optimizer_state']['state'][key][param_key] 
                                for result in results if 'state' in result['optimizer_state']
                                and key in result['optimizer_state']['state']
                                and param_key in result['optimizer_state']['state'][key]
                            ) / num_agents
            self._optimizer_state = avg_optimizer
            
            # Average scheduler states
            avg_scheduler = results[0]['scheduler_state'].copy()
            for key, value in avg_scheduler.items():
                if isinstance(value, torch.Tensor):
                    avg_scheduler[key] = sum(
                        result['scheduler_state'][key] 
                        for result in results if key in result['scheduler_state']
                    ) / num_agents
            self._scheduler_state = avg_scheduler
            
            # Update episode count (max of all agents)
            self._episode_count = max(result['episode_count'] for result in results)
            self._last_update = time.time()
            
            # Clear completed agents
            self._completed_agents.clear()
            
            print(f"Averaged parameters from {num_agents} agents at episode {self._episode_count}")
            return True
    
    def save_to_checkpoint(self, checkpoint_path: str, config: Dict) -> None:
        """Save current shared state to disk checkpoint"""
        from PoliwhiRL.environment import PyBoyEnvironment as Env
        from PoliwhiRL.agents.PPO import PPOAgent
        
        os.makedirs(checkpoint_path, exist_ok=True)
        
        with self.lock:
            # Create temporary agent for saving
            temp_env = Env.create_with_shared_temp(config)
            try:
                temp_agent = PPOAgent(
                    temp_env.output_shape(),
                    temp_env.action_space.n,
                    config
                )
                
                # Apply shared parameters to temp agent
                self.update_agent_model(temp_agent, -1)  # -1 for temp agent
                
                # Save components
                torch.save(
                    temp_agent.model.actor_critic.state_dict(), 
                    f"{checkpoint_path}/actor_critic.pth"
                )
                torch.save(
                    temp_agent.model.optimizer.state_dict(), 
                    f"{checkpoint_path}/optimizer.pth"
                )
                torch.save(
                    temp_agent.model.scheduler.state_dict(), 
                    f"{checkpoint_path}/scheduler.pth"
                )
                temp_agent.model.icm.save(f"{checkpoint_path}/icm")
                
                # Save metadata
                info = {
                    "episode": self._episode_count,
                    "best_reward": 0,
                    "episode_data": {},
                    "shared_memory_timestamp": self._last_update
                }
                torch.save(info, f"{checkpoint_path}/info.pth")
                
                print(f"Shared model saved to {checkpoint_path}")
                
            finally:
                temp_env.close()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status for monitoring"""
        with self.lock:
            return {
                'episode_count': self._episode_count,
                'last_update': self._last_update,
                'active_agents': len(self._active_agents),
                'completed_agents': len(self._completed_agents),
                'param_groups': list(self._current_params.keys()),
                'history_size': len(self._parameter_history)
            }
    
    def cleanup(self) -> None:
        """Clean up shared memory resources"""
        with self.lock:
            self._current_params.clear()
            self._optimizer_state.clear()
            self._scheduler_state.clear()
            self._active_agents.clear()
            self._completed_agents.clear()
            self._parameter_history.clear()
            print("Shared memory manager cleaned up")


# Global instance for shared access
_shared_memory_manager: Optional[SharedMemoryModelManager] = None


def get_shared_memory_manager(device: str = "cpu") -> SharedMemoryModelManager:
    """Get or create global shared memory manager"""
    global _shared_memory_manager
    if _shared_memory_manager is None:
        _shared_memory_manager = SharedMemoryModelManager(device)
    return _shared_memory_manager


def reset_shared_memory_manager() -> None:
    """Reset global shared memory manager"""
    global _shared_memory_manager
    if _shared_memory_manager:
        _shared_memory_manager.cleanup()
    _shared_memory_manager = None