# -*- coding: utf-8 -*-
import os
import torch
import torch.multiprocessing as mp
from PoliwhiRL.environment import PyBoyEnvironment as Env
from PoliwhiRL.agents.PPO import PPOAgent
from PoliwhiRL.models.PPO import PPOTransformer
from PoliwhiRL.models.ICM import ICMModule
from tqdm.auto import tqdm as auto_tqdm


def run_single_agent(
    agent_id, config, shared_model_state, icm_state, iteration, base_checkpoint, cumulative_episodes
):
    """
    Run a single PPO agent training episode.
    This function runs in a separate process.
    """
    # Update config for this specific agent
    agent_config = config.copy()
    agent_config["checkpoint"] = f"{base_checkpoint}_{agent_id}"
    agent_config["record_path"] = f"{config['record_path']}_{agent_id}"
    agent_config["export_state_loc"] = f"{config['export_state_loc']}_{agent_id}"
    agent_config["results_dir"] = f"{config['results_dir']}_{agent_id}"
    agent_config["record"] = False if agent_id != 0 else config["record"]
    agent_config["report_episode"] = False  # Disable tqdm in subprocess
    # Set the starting episode number to continue from previous iterations
    agent_config["start_episode"] = cumulative_episodes
    # Set the worker ID for debug prints
    agent_config["tqdm_worker_id"] = agent_id
    
    print(f"Agent {agent_id} starting (iteration {iteration}, starting from episode {cumulative_episodes})")
    
    # Create environment
    env = Env(agent_config)
    try:
        state_shape = env.output_shape()
        num_actions = env.action_space.n
        
        # Create agent
        agent = PPOAgent(state_shape, num_actions, agent_config)
        
        # Load shared model state if provided
        if shared_model_state is not None:
            agent.model.actor_critic.load_state_dict(shared_model_state)
        if icm_state is not None:
            agent.model.icm.icm.load_state_dict(icm_state)
        
        # Load agent's own checkpoint to preserve its episode data
        agent_checkpoint = agent_config["checkpoint"]
        if os.path.exists(agent_checkpoint):
            print(f"Agent {agent_id} loading its own checkpoint: {agent_checkpoint}")
            agent.load_model(agent_checkpoint)
        
        # Only load curriculum checkpoint if we don't have a shared model state
        # (This handles the case where the first iteration failed to load the curriculum checkpoint)
        if shared_model_state is None and agent_config.get("load_checkpoint") and agent_config["load_checkpoint"] != "":
            curriculum_checkpoint = agent_config["load_checkpoint"]
            if os.path.exists(curriculum_checkpoint):
                print(f"Agent {agent_id} loading curriculum checkpoint: {curriculum_checkpoint}")
                agent.load_model(curriculum_checkpoint)
        
        # Run training
        if agent_config["use_curriculum"]:
            agent.run_curriculum(
                1, agent_config["N_goals_target"], agent_config["N_goals_increment"]
            )
        else:
            agent.train_agent()
        
        print(f"Agent {agent_id} completed successfully")
        
        # Return model state dicts for averaging (move to CPU for serialization)
        actor_critic_state = agent.model.actor_critic.state_dict()
        icm_state = agent.model.icm.icm.state_dict()
        optimizer_state = agent.model.optimizer.state_dict()
        scheduler_state = agent.model.scheduler.state_dict()
        
        # Move all tensors to CPU
        actor_critic_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in actor_critic_state.items()}
        icm_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in icm_state.items()}
        
        # Handle optimizer state dict which has nested structure
        optimizer_cpu = optimizer_state.copy()
        if 'state' in optimizer_cpu:
            for key, state in optimizer_cpu['state'].items():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        optimizer_cpu['state'][key][k] = v.cpu()
        
        return {
            "actor_critic": actor_critic_cpu,
            "icm": icm_cpu,
            "optimizer": optimizer_cpu,
            "scheduler": scheduler_state,  # scheduler state usually doesn't have tensors
            "episode": agent.episode,
            "checkpoint_path": agent_config["checkpoint"]
        }
        
    except Exception as e:
        print(f"Agent {agent_id} failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        env.close()


class PPOParallelRunner:
    """
    Simple parallel runner for PPO agents using multiprocessing.Pool.
    Based on the DQN ParallelAgentRunner pattern.
    """
    
    def __init__(self, config):
        self.config = config
        self.num_agents = config["ppo_num_agents"]
        self.iterations = config["ppo_iterations"]
        self.base_checkpoint = config["checkpoint"]
        self.total_episodes_run = config["start_episode"]
        
        # Set multiprocessing start method
        if mp.get_start_method(allow_none=True) != "spawn":
            try:
                mp.set_start_method("spawn", force=True)
            except RuntimeError:
                pass
    
    def get_shared_model_state(self, iteration):
        """Get the current shared model state if it exists"""
        # For iteration 0, try to load from the curriculum checkpoint first
        if iteration == 0 and self.config.get("load_checkpoint") and self.config["load_checkpoint"] != "":
            curriculum_checkpoint = self.config["load_checkpoint"]
            if os.path.exists(curriculum_checkpoint):
                try:
                    print(f"Loading initial shared model from curriculum checkpoint: {curriculum_checkpoint}")
                    device = torch.device("cpu")
                    actor_critic_state = torch.load(
                        f"{curriculum_checkpoint}/actor_critic.pth",
                        map_location=device,
                        weights_only=True
                    )
                    icm_state = torch.load(
                        f"{curriculum_checkpoint}/icm_icm.pth",
                        map_location=device,
                        weights_only=True
                    )
                    return actor_critic_state, icm_state
                except Exception as e:
                    print(f"Could not load curriculum checkpoint: {e}")
        
        # For subsequent iterations or if no curriculum checkpoint, use current stage checkpoint
        if os.path.exists(self.base_checkpoint):
            try:
                print(f"Loading shared model from current stage: {self.base_checkpoint}")
                device = torch.device("cpu")
                actor_critic_state = torch.load(
                    f"{self.base_checkpoint}/actor_critic.pth",
                    map_location=device,
                    weights_only=True
                )
                icm_state = torch.load(
                    f"{self.base_checkpoint}/icm_icm.pth",
                    map_location=device,
                    weights_only=True
                )
                return actor_critic_state, icm_state
            except Exception as e:
                print(f"Could not load shared model state: {e}")
                return None, None
        return None, None
    
    def train(self):
        """Main training loop"""
        print(f"Starting PPO parallel training with {self.num_agents} agents")
        print(f"Running {self.iterations} iterations")
        
        # Get initial environment shape
        env = Env(self.config)
        try:
            state_shape = env.output_shape()
            num_actions = env.action_space.n
        finally:
            env.close()
        
        # Training loop
        for iteration in auto_tqdm(range(self.iterations), desc="Iterations"):
            print(f"\n{'='*60}")
            print(f"Starting iteration {iteration + 1}/{self.iterations}")
            print(f"{'='*60}")
            
            # Get current shared model state
            shared_model_state, icm_state = self.get_shared_model_state(iteration)
            
            # Run agents in parallel
            with mp.Pool(processes=self.num_agents) as pool:
                results = pool.starmap(
                    run_single_agent,
                    [(i, self.config, shared_model_state, icm_state, iteration, self.base_checkpoint, self.total_episodes_run) 
                     for i in range(self.num_agents)]
                )
            
            # Filter out failed agents
            successful_results = [r for r in results if r is not None]
            
            if not successful_results:
                print("WARNING: All agents failed in this iteration!")
                continue
            
            print(f"\n{len(successful_results)}/{self.num_agents} agents completed successfully")
            
            # Update episode count
            self.total_episodes_run += self.config["num_episodes"]
            
            # Average models and save
            self.average_and_save_models(successful_results, state_shape, num_actions)
            
            print(f"\nIteration {iteration + 1} completed. Total episodes: {self.total_episodes_run}")
        
        print("\nTraining completed!")
        
    def average_and_save_models(self, results, state_shape, num_actions):
        """Average model parameters from successful agents and save to checkpoint"""
        print("Averaging models from successful agents...")
        
        # Create a temporary agent to hold averaged model
        temp_agent = PPOAgent(state_shape, num_actions, self.config)
        device = temp_agent.device
        
        # Average actor_critic parameters
        actor_critic_params = {}
        for name, param in temp_agent.model.actor_critic.named_parameters():
            averaged_param = torch.zeros_like(param)
            for result in results:
                # Move tensor to correct device if needed
                param_tensor = result["actor_critic"][name]
                if param_tensor.device != device:
                    param_tensor = param_tensor.to(device)
                averaged_param += param_tensor
            averaged_param /= len(results)
            actor_critic_params[name] = averaged_param
        
        # Apply averaged parameters
        for name, param in temp_agent.model.actor_critic.named_parameters():
            param.data.copy_(actor_critic_params[name])
        
        # Average ICM parameters
        icm_params = {}
        for name, param in temp_agent.model.icm.icm.named_parameters():
            averaged_param = torch.zeros_like(param)
            for result in results:
                # Move tensor to correct device if needed
                param_tensor = result["icm"][name]
                if param_tensor.device != device:
                    param_tensor = param_tensor.to(device)
                averaged_param += param_tensor
            averaged_param /= len(results)
            icm_params[name] = averaged_param
        
        # Apply averaged ICM parameters
        for name, param in temp_agent.model.icm.icm.named_parameters():
            param.data.copy_(icm_params[name])
        
        # Average optimizer state (first agent's structure, averaged values)
        if results[0]["optimizer"]["state"]:
            optimizer_state = results[0]["optimizer"].copy()
            for key in optimizer_state["state"]:
                for param_key in optimizer_state["state"][key]:
                    if isinstance(optimizer_state["state"][key][param_key], torch.Tensor):
                        # Get first tensor to determine shape/device
                        first_tensor = optimizer_state["state"][key][param_key]
                        averaged_value = torch.zeros_like(first_tensor).to(device)
                        for result in results:
                            if key in result["optimizer"]["state"]:
                                param_tensor = result["optimizer"]["state"][key][param_key]
                                if param_tensor.device != device:
                                    param_tensor = param_tensor.to(device)
                                averaged_value += param_tensor
                        averaged_value /= len(results)
                        optimizer_state["state"][key][param_key] = averaged_value
            temp_agent.model.optimizer.load_state_dict(optimizer_state)
        
        # Average scheduler state
        scheduler_state = results[0]["scheduler"].copy()
        for key, value in scheduler_state.items():
            if isinstance(value, (int, float)) and key != "last_epoch":
                averaged_value = sum(r["scheduler"][key] for r in results) / len(results)
                scheduler_state[key] = averaged_value
        temp_agent.model.scheduler.load_state_dict(scheduler_state)
        
        # Set episode count
        temp_agent.episode = self.total_episodes_run
        
        # Save to checkpoint
        print(f"Saving averaged model to {self.base_checkpoint}")
        temp_agent.save_model(self.base_checkpoint)
        
        # Keep individual agent checkpoints to preserve episode data across iterations
        print(f"Keeping {len(results)} individual agent checkpoints for episode data persistence")