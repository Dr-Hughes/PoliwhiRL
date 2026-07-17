# -*- coding: utf-8 -*-
import os


def _resolve_auto_resume(config):
    """Continue from the latest checkpoint already in this run's output
    folder unless one was explicitly requested.

    Makes start/stop/append training the DEFAULT: re-running the same config
    picks up the most recent weights (the main ``Checkpoints/`` dir, which is
    overwritten every ``checkpoint_frequency`` — i.e. the latest, not the
    best-so-far ``Checkpoints/best/``) and continues, so you can stop a run
    and add more rollouts without re-pointing ``load_checkpoint`` by hand.

    Only fires when ``load_checkpoint`` is empty (an explicit path always
    wins) AND a resumable checkpoint actually exists. A brand-new run's empty
    output folder simply starts from scratch. ``erase: true`` wipes the
    folder first (handled in main), so it also correctly starts fresh.
    """
    if config.get("load_checkpoint"):
        return
    ckpt_dir = config.get("checkpoint")
    if not ckpt_dir:
        return
    if os.path.isfile(os.path.join(ckpt_dir, "actor_critic.pth")) and os.path.isfile(
        os.path.join(ckpt_dir, "info.pth")
    ):
        config["load_checkpoint"] = ckpt_dir
        print(f"[PPO] Auto-resuming from latest checkpoint: {ckpt_dir}")


def _resolve_auto_gamma(config):
    """``ppo_gamma: "auto"`` derives the discount from the episode budget:
    gamma = 1 - 4/episode_length, capped at 0.999 (a ~1000-step credit
    horizon), matching the 0.998–0.999 the game-beating Pokémon-RL runs use.

    The previous cap of 0.995 pinned the horizon at ~200 steps regardless of
    episode length. Milestones (leave house -> starter -> route -> next town)
    are typically hundreds-to-thousands of steps apart, so at 0.995 their
    discounted value at decision time was ~0 and the critic simply learned to
    predict the dense exploration return, never the directed milestones — the
    single biggest structural barrier to progressing through the game rather
    than wandering (2026-07-16 diagnosis)."""
    gamma = config.get("ppo_gamma", 0.99)
    if isinstance(gamma, str):
        ep_len = int(config["episode_length"])
        config["ppo_gamma"] = min(0.999, 1.0 - 4.0 / max(ep_len, 8))
        print(
            f"[PPO] ppo_gamma=auto resolved to {config['ppo_gamma']:.5f} "
            f"(episode_length={ep_len})"
        )


def setup_and_train_PPO(config):
    from PoliwhiRL.environment import PyBoyEnvironment as Env
    from PoliwhiRL.agents.PPO import VecPPOAgent

    _resolve_auto_resume(config)
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
