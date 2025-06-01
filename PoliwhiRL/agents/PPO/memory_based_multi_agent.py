# -*- coding: utf-8 -*-
"""
Memory-based multi-agent PPO implementation that minimizes disk I/O
by keeping model parameters in shared memory and only persisting
final checkpoints between stages.
"""

import torch
import torch.multiprocessing as mp
import os
import time
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from PoliwhiRL.environment import PyBoyEnvironment as Env
from PoliwhiRL.agents.PPO import PPOAgent
from PoliwhiRL.utils.resource_manager import get_resource_pool
import numpy as np


@dataclass
class SharedModelState:
    """Container for shared model parameters in memory"""
    actor_critic_params: Dict[str, torch.Tensor]
    icm_params: Dict[str, torch.Tensor]
    optimizer_state: Dict
    scheduler_state: Dict
    episode_count: int
    lock: threading.RLock


class MemoryBasedMultiAgent:
    """
    Multi-agent PPO that maintains model parameters in shared memory,
    only writing to disk for stage transitions and final checkpoints.
    """
    
    def __init__(self, config):
        self.config = config
        self.num_agents = config["ppo_num_agents"]
        self.iterations = config["ppo_iterations"]
        self.total_episodes_run = config["start_episode"]
        self.device = torch.device(config.get("device", "cpu"))
        
        # Shared memory for model parameters
        self.shared_model_state: Optional[SharedModelState] = None
        self.agent_results_queue = mp.Queue()
        self.resource_pool = get_resource_pool()
        
        # Initialize shared memory manager
        if mp.get_start_method(allow_none=True) != "spawn":
            try:
                mp.set_start_method("spawn", force=True)
            except RuntimeError:
                # Start method already set
                pass
        self.manager = mp.Manager()
        
    def initialize_shared_model(self, state_shape: Tuple, num_actions: int) -> None:
        """Initialize shared model state in memory"""
        # Create a reference agent to get model structure
        temp_env = Env.create_with_shared_temp(self.config)
        try:
            reference_agent = PPOAgent(state_shape, num_actions, self.config)
            
            # Load curriculum checkpoint first if specified (for stage transitions)
            curriculum_checkpoint = self.config.get("load_checkpoint", "")
            if curriculum_checkpoint and os.path.exists(curriculum_checkpoint):
                reference_agent.load_model(curriculum_checkpoint)
                print(f"Loaded curriculum checkpoint from {curriculum_checkpoint}")
            
            # Load existing checkpoint if available (for resuming within stage)
            checkpoint_path = self.config["checkpoint"]
            if os.path.exists(checkpoint_path):
                reference_agent.load_model(checkpoint_path)
                print(f"Loaded existing checkpoint from {checkpoint_path}")
            
            # Extract model parameters to shared memory
            self.shared_model_state = SharedModelState(
                actor_critic_params=self._extract_params(reference_agent.model.actor_critic),
                icm_params=self._extract_params(reference_agent.model.icm.icm),
                optimizer_state=reference_agent.model.optimizer.state_dict(),
                scheduler_state=reference_agent.model.scheduler.state_dict(),
                episode_count=reference_agent.episode,
                lock=threading.RLock()
            )
            
            print("Shared model state initialized in memory")
            
        finally:
            temp_env.close()
    
    def _extract_params(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """Extract model parameters as tensors"""
        return {name: param.data.clone() for name, param in model.named_parameters()}
    
    def _update_model_from_shared(self, agent: PPOAgent) -> None:
        """Update agent model from shared memory state"""
        with self.shared_model_state.lock:
            # Update actor-critic parameters
            for name, param in agent.model.actor_critic.named_parameters():
                if name in self.shared_model_state.actor_critic_params:
                    param.data.copy_(self.shared_model_state.actor_critic_params[name])
            
            # Update ICM parameters
            for name, param in agent.model.icm.icm.named_parameters():
                if name in self.shared_model_state.icm_params:
                    param.data.copy_(self.shared_model_state.icm_params[name])
            
            # Update optimizer and scheduler states
            agent.model.optimizer.load_state_dict(self.shared_model_state.optimizer_state)
            agent.model.scheduler.load_state_dict(self.shared_model_state.scheduler_state)
            agent.episode = self.shared_model_state.episode_count
    
    def _collect_agent_results(self, agent: PPOAgent) -> Dict[str, torch.Tensor]:
        """Collect trained model parameters from agent"""
        return {
            'actor_critic': self._extract_params(agent.model.actor_critic),
            'icm': self._extract_params(agent.model.icm.icm),
            'optimizer': agent.model.optimizer.state_dict(),
            'scheduler': agent.model.scheduler.state_dict(),
            'episode_count': agent.episode
        }
    
    def _average_agent_results(self, agent_results: List[Dict]) -> None:
        """Average results from all agents and update shared state"""
        if not agent_results:
            return
            
        num_agents = len(agent_results)
        
        with self.shared_model_state.lock:
            # Average actor-critic parameters
            for param_name in self.shared_model_state.actor_critic_params:
                averaged_param = sum(
                    result['actor_critic'][param_name] for result in agent_results
                ) / num_agents
                self.shared_model_state.actor_critic_params[param_name].copy_(averaged_param)
            
            # Average ICM parameters
            for param_name in self.shared_model_state.icm_params:
                averaged_param = sum(
                    result['icm'][param_name] for result in agent_results
                ) / num_agents
                self.shared_model_state.icm_params[param_name].copy_(averaged_param)
            
            # Average optimizer states (only numerical values)
            avg_optimizer = agent_results[0]['optimizer']
            for key in avg_optimizer.get('state', {}):
                for param_key in avg_optimizer['state'][key]:
                    if isinstance(avg_optimizer['state'][key][param_key], torch.Tensor):
                        avg_optimizer['state'][key][param_key] = sum(
                            result['optimizer']['state'][key][param_key] 
                            for result in agent_results
                        ) / num_agents
            self.shared_model_state.optimizer_state = avg_optimizer
            
            # Average scheduler states
            avg_scheduler = agent_results[0]['scheduler']
            for key, value in avg_scheduler.items():
                if isinstance(value, torch.Tensor):
                    avg_scheduler[key] = sum(
                        result['scheduler'][key] for result in agent_results
                    ) / num_agents
            self.shared_model_state.scheduler_state = avg_scheduler
            
            # Update episode count
            self.shared_model_state.episode_count = self.total_episodes_run
    
    @staticmethod
    def train_agent_subprocess(agent_id: int, agent_config: Dict, results_queue: mp.Queue, 
                             shared_state_data: Optional[Dict] = None) -> None:
        """Run single agent training in subprocess"""
        import signal
        import time
        
        def signal_handler(signum, frame):
            print(f"Agent {agent_id} received signal {signum}, cleaning up...")
            exit(1)
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        print(f"Starting agent {agent_id} training (PID: {os.getpid()})")
        
        env = None
        agent = None
        
        try:
            # Create environment and agent in subprocess with retry logic
            env = None
            max_env_retries = 3
            for retry in range(max_env_retries):
                try:
                    env = Env(agent_config, shared_temp_dir=agent_config.get('shared_temp_dir'))
                    break
                except Exception as e:
                    print(f"Agent {agent_id} environment creation attempt {retry + 1} failed: {e}")
                    if retry == max_env_retries - 1:
                        raise
                    time.sleep(1)  # Wait before retry
            
            agent = PPOAgent(
                env.output_shape(), 
                env.action_space.n, 
                agent_config
            )
            
            # Load curriculum checkpoint if specified
            curriculum_checkpoint = agent_config.get("load_checkpoint", "")
            if curriculum_checkpoint and os.path.exists(curriculum_checkpoint):
                agent.load_model(curriculum_checkpoint)
                print(f"Agent {agent_id} loaded curriculum checkpoint from {curriculum_checkpoint}")
            
            # Apply shared state if provided
            if shared_state_data:
                MemoryBasedMultiAgent._apply_shared_state_to_agent(agent, shared_state_data)
                print(f"Agent {agent_id} synchronized with shared model state")
            
            # Run training with timeout protection
            training_start_time = time.time()
            max_training_time = agent_config.get("max_training_time", 3600)  # 1 hour default
            
            if agent_config.get("use_curriculum", False):
                agent.run_curriculum(
                    1, agent_config["N_goals_target"], agent_config.get("N_goals_increment", 1)
                )
            else:
                agent.train_agent()
            
            training_duration = time.time() - training_start_time
            print(f"Agent {agent_id} training completed in {training_duration:.2f} seconds")
            
            # Collect results - ensure all tensors are on CPU for multiprocessing
            def move_state_dict_to_cpu(state_dict):
                """Recursively move all tensors in a state dict to CPU"""
                result = {}
                for k, v in state_dict.items():
                    if isinstance(v, torch.Tensor):
                        result[k] = v.cpu()
                    elif isinstance(v, dict):
                        result[k] = move_state_dict_to_cpu(v)
                    else:
                        result[k] = v
                return result
            
            results = {
                'actor_critic': {name: param.data.clone().cpu() for name, param in agent.model.actor_critic.named_parameters()},
                'icm': {name: param.data.clone().cpu() for name, param in agent.model.icm.icm.named_parameters()},
                'optimizer': move_state_dict_to_cpu(agent.model.optimizer.state_dict()),
                'scheduler': move_state_dict_to_cpu(agent.model.scheduler.state_dict()),
                'episode_count': agent.episode
            }
            
            # Send results back
            results_queue.put((agent_id, results))
            
            print(f"Agent {agent_id} completed training successfully")
            
        except Exception as e:
            print(f"Agent {agent_id} failed: {e}")
            import traceback
            traceback.print_exc()
            results_queue.put((agent_id, None))
        finally:
            # Proper cleanup order
            if agent:
                try:
                    del agent
                except:
                    pass
            if env:
                try:
                    env.close()
                except:
                    pass
            # Force garbage collection to clean up PyBoy resources
            import gc
            gc.collect()
    
    @staticmethod
    def _apply_shared_state_to_agent(agent, shared_state_data: Dict) -> None:
        """Apply shared state data to an agent"""
        try:
            # Update actor-critic parameters
            if 'actor_critic_params' in shared_state_data:
                for name, param in agent.model.actor_critic.named_parameters():
                    if name in shared_state_data['actor_critic_params']:
                        param.data.copy_(shared_state_data['actor_critic_params'][name])
            
            # Update ICM parameters
            if 'icm_params' in shared_state_data:
                for name, param in agent.model.icm.icm.named_parameters():
                    if name in shared_state_data['icm_params']:
                        param.data.copy_(shared_state_data['icm_params'][name])
            
            # Update optimizer and scheduler states
            if 'optimizer_state' in shared_state_data:
                agent.model.optimizer.load_state_dict(shared_state_data['optimizer_state'])
            
            if 'scheduler_state' in shared_state_data:
                agent.model.scheduler.load_state_dict(shared_state_data['scheduler_state'])
            
            if 'episode_count' in shared_state_data:
                agent.episode = shared_state_data['episode_count']
                
        except Exception as e:
            print(f"Warning: Could not fully apply shared state: {e}")
    
    def run_parallel_iteration(self, iteration: int) -> None:
        """Run one iteration of parallel training"""
        print(f"Starting iteration {iteration + 1}/{self.iterations}")
        
        processes = []
        agent_configs = []
        
        # Set up shared temp directory - keep env alive until all processes complete
        temp_env = Env.create_with_shared_temp(self.config)
        shared_temp_dir = temp_env.temp_dir
        
        # Prepare agent configurations
        for i in range(self.num_agents):
            agent_config = self.config.copy()
            agent_config.update({
                # No individual checkpoints - only shared memory and final stage checkpoint
                "checkpoint": None,  # Disable individual agent checkpoints
                "record": False if i != 0 else self.config["record"],
                "record_path": f"{self.config['record_path']}_{i}",
                "results_dir": self.config["results_dir"],  # All agents use same results folder
                "export_state_loc": f"{self.config['export_state_loc']}_{i}",
                "tqdm_position": i,
                "tqdm_desc_prefix": f"Agent {i}",
                "tqdm_worker_id": i,
                "report_episode": False,  # Disable in subprocess
                "shared_temp_dir": shared_temp_dir,
                "save_checkpoint": False  # Disable checkpoint saving for individual agents
            })
            agent_configs.append(agent_config)
        
        # Get current shared state for agents - ensure tensors are on CPU for multiprocessing
        shared_state_data = None
        if self.shared_model_state and iteration > 0:
            with self.shared_model_state.lock:
                def move_state_dict_to_cpu(state_dict):
                    """Recursively move all tensors in a state dict to CPU"""
                    result = {}
                    for k, v in state_dict.items():
                        if isinstance(v, torch.Tensor):
                            result[k] = v.cpu()
                        elif isinstance(v, dict):
                            result[k] = move_state_dict_to_cpu(v)
                        else:
                            result[k] = v
                    return result
                
                shared_state_data = {
                    'actor_critic_params': {k: v.clone().cpu() for k, v in self.shared_model_state.actor_critic_params.items()},
                    'icm_params': {k: v.clone().cpu() for k, v in self.shared_model_state.icm_params.items()},
                    'optimizer_state': move_state_dict_to_cpu(self.shared_model_state.optimizer_state),
                    'scheduler_state': move_state_dict_to_cpu(self.shared_model_state.scheduler_state),
                    'episode_count': self.shared_model_state.episode_count
                }
        
        # Start agent processes
        for i, config in enumerate(agent_configs):
            process = mp.Process(
                target=MemoryBasedMultiAgent.train_agent_subprocess,
                args=(i, config, self.agent_results_queue, shared_state_data)
            )
            process.start()
            processes.append(process)
        
        # Wait for all agents to complete using a non-blocking approach
        try:
            import time
            timeout_per_agent = self.config.get("agent_timeout", 300)
            start_time = time.time()
            check_interval = 1  # Check every second
            
            while True:
                # Check which processes are still alive
                alive_processes = [(i, p) for i, p in enumerate(processes) if p.is_alive()]
                
                if not alive_processes:
                    print("All agent processes completed")
                    break
                
                # Check for timeouts
                elapsed = time.time() - start_time
                if elapsed > timeout_per_agent:
                    print(f"Timeout reached ({timeout_per_agent}s), terminating remaining processes")
                    for i, process in alive_processes:
                        if process.is_alive():
                            print(f"Terminating unresponsive agent {i} process")
                            process.terminate()
                    
                    # Give processes 5 seconds to terminate gracefully
                    time.sleep(5)
                    
                    # Force kill any still-alive processes
                    for i, process in alive_processes:
                        if process.is_alive():
                            print(f"Force killing agent {i} process")
                            process.kill()
                    break
                
                # Print status every 30 seconds
                if int(elapsed) % 30 == 0:
                    print(f"Waiting for {len(alive_processes)} agents to complete... ({elapsed:.0f}s elapsed)")
                
                time.sleep(check_interval)
                
        finally:
            # Ensure all processes are cleaned up
            for i, process in enumerate(processes):
                if process.is_alive():
                    try:
                        process.terminate()
                        process.join(2)
                        if process.is_alive():
                            process.kill()
                    except Exception:
                        pass
            
            # Close the shared temp environment after all processes complete
            try:
                temp_env.close()
            except Exception as e:
                print(f"Warning: Error closing shared temp environment: {e}")
        
        # Collect results from all agents
        agent_results = []
        successful_agents = 0
        
        # Count successful agents by checking queue size or completed processes
        completed_agents = sum(1 for p in processes if not p.is_alive())
        print(f"Collecting results from {completed_agents} completed agents")
        
        for _ in range(self.num_agents):
            try:
                agent_id, result = self.agent_results_queue.get(timeout=5)
                if result is not None:
                    agent_results.append(result)
                    successful_agents += 1
                    print(f"Collected result from agent {agent_id}")
                else:
                    print(f"Agent {agent_id} returned None result")
            except Exception as e:
                print(f"Warning: Failed to collect result from agent: {e}")
                break  # Stop trying if queue is empty
        
        print(f"Successfully collected {successful_agents} agent results")
        
        # Average results and update shared state
        if agent_results:
            self._average_agent_results(agent_results)
            self.total_episodes_run += self.config["num_episodes"]
            print(f"Averaged {len(agent_results)} agent results")
        else:
            print("Warning: No valid agent results to average")
    
    def save_checkpoint_to_disk(self, checkpoint_path: str) -> None:
        """Save current shared model state to disk (only for stage transitions)"""
        if self.shared_model_state is None:
            return
            
        os.makedirs(checkpoint_path, exist_ok=True)
        
        with self.shared_model_state.lock:
            # Create temporary agent to save in correct format
            temp_env = Env.create_with_shared_temp(self.config)
            try:
                temp_agent = PPOAgent(
                    temp_env.output_shape(),
                    temp_env.action_space.n,
                    self.config
                )
                
                # Apply shared state to temp agent
                self._update_model_from_shared(temp_agent)
                
                # Save using existing save mechanism
                torch.save(temp_agent.model.actor_critic.state_dict(), f"{checkpoint_path}/actor_critic.pth")
                torch.save(temp_agent.model.optimizer.state_dict(), f"{checkpoint_path}/optimizer.pth")
                torch.save(temp_agent.model.scheduler.state_dict(), f"{checkpoint_path}/scheduler.pth")
                temp_agent.model.icm.save(f"{checkpoint_path}/icm")
                
                # Save episode info
                info = {
                    "episode": self.shared_model_state.episode_count,
                    "best_reward": 0,
                    "episode_data": {}
                }
                torch.save(info, f"{checkpoint_path}/info.pth")
                
                print(f"Checkpoint saved to {checkpoint_path}")
                
            finally:
                temp_env.close()
    
    def train(self) -> torch.nn.Module:
        """Main training loop with memory-based agent management"""
        env = Env.create_with_shared_temp(self.config)
        try:
            # Initialize shared model state
            self.initialize_shared_model(env.output_shape(), env.action_space.n)
            
            print(f"Starting memory-based multi-agent training with {self.num_agents} agents")
            print("Model parameters maintained in shared memory")
            
            # Training iterations
            for iteration in range(self.iterations):
                self.run_parallel_iteration(iteration)
                print(f"Iteration {iteration + 1} completed. Episodes: {self.total_episodes_run}")
            
            # Save final averaged checkpoint for stage transition only
            print(f"Training completed. Saving final checkpoint for curriculum stage transition...")
            self.save_checkpoint_to_disk(self.config["checkpoint"])
            
            return self.shared_model_state
            
        finally:
            env.close()