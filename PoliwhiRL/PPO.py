# -*- coding: utf-8 -*-
def _resolve_auto_gamma(config):
    """``ppo_gamma: "auto"`` derives the discount from the episode budget:
    gamma = 1 - 4/episode_length (capped at 0.9995), i.e. an effective
    credit horizon of ~episode_length/4 steps. A fixed 0.99 gave a ~100-step
    horizon — invisible across the multi-hundred-step gaps between
    curriculum goals, so the value function never saw the next rung."""
    gamma = config.get("ppo_gamma", 0.99)
    if isinstance(gamma, str):
        ep_len = int(config["episode_length"])
        config["ppo_gamma"] = min(0.9995, 1.0 - 4.0 / max(ep_len, 8))
        print(
            f"[PPO] ppo_gamma=auto resolved to {config['ppo_gamma']:.5f} "
            f"(episode_length={ep_len})"
        )


def setup_and_train_PPO(config):
    from PoliwhiRL.environment import PyBoyEnvironment as Env
    from PoliwhiRL.agents.PPO import VecPPOAgent

    _resolve_auto_gamma(config)

    # Probe shape/action size with a single short-lived env. The vec wrapper
    # spins up its own worker processes — there is ONE training
    # implementation; num_envs == 1 just runs a single worker (as a pure
    # probe, no snapshot seeding).
    env = Env(config)
    try:
        state_shape = env.output_shape()
        ram_obs_dim = env.ram_observation_shape()[0]
        num_actions = env.action_space.n
    finally:
        env.close()

    # Make the RAM dim available to downstream constructors (model, buffers,
    # agents) without requiring them to call env again.
    config["ram_obs_dim"] = ram_obs_dim

    agent = VecPPOAgent(state_shape, num_actions, config)

    if config["load_checkpoint"]:
        agent.load_model(config["load_checkpoint"])

    agent.train_agent()
    return agent.model
