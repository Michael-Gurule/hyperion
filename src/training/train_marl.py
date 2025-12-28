"""
Multi-agent reinforcement learning training script.
Uses RLlib with PPO for swarm coordination.
"""

import os
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
import yaml
from typing import Dict, Any

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.env.rllib_wrapper import RLlibHyperionEnv
from src.utils.logger import setup_logger
from src.training.curriculum import CurriculumScheduler


logger = setup_logger("hyperion_training")


def env_creator(env_config):
    """Create environment for RLlib."""
    return RLlibHyperionEnv(env_config)


def train_swarm_policy(
    config_path: str = "config.yaml",
    num_iterations: int = 100,
    checkpoint_dir: str = "./checkpoints",
):
    """
    Train swarm coordination policy using multi-agent PPO.

    Args:
        config_path: Path to configuration file
        num_iterations: Number of training iterations
        checkpoint_dir: Directory to save checkpoints
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    env_config = config["environment"]
    training_config = config["training"]
    ppo_config = training_config["ppo"]

    logger.info("Starting HYPERION training")
    logger.info(f"Environment config: {env_config}")

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Register environment
    register_env("hyperion_swarm", lambda config: env_creator(config))

    # Create dummy environment to get spaces
    dummy_env = env_creator(env_config)
    agent_ids = list(dummy_env.get_agent_ids())
    obs_space = dummy_env.observation_space[
        agent_ids[0]
    ]  # Use dict access instead of method
    act_space = dummy_env.action_space[
        agent_ids[0]
    ]  # Use dict access instead of method

    # Configure PPO
    ppo = (
        PPOConfig()
        .environment("hyperion_swarm", env_config=env_config)
        .framework("torch")
        .resources(num_gpus=training_config.get("num_gpus", 0), num_cpus_per_worker=1)
        .env_runners(
            num_env_runners=training_config.get("num_workers", 4),
            num_envs_per_env_runner=training_config.get("num_envs_per_worker", 1),
            rollout_fragment_length=training_config.get("rollout_fragment_length", 200),
        )
        .training(
            lr=ppo_config["lr"],
            gamma=ppo_config["gamma"],
            lambda_=ppo_config["lambda_gae"],
            clip_param=ppo_config["clip_param"],
            vf_clip_param=ppo_config["vf_clip_param"],
            entropy_coeff=ppo_config["entropy_coeff"],
            num_epochs=ppo_config["num_sgd_iter"],  # Changed parameter name
            minibatch_size=ppo_config["sgd_minibatch_size"],  # Changed parameter name
            train_batch_size=ppo_config["train_batch_size"],
        )
        .multi_agent(
            policies={"shared_policy": (None, obs_space, act_space, {})},
            policy_mapping_fn=lambda agent_id,
            episode,
            worker,
            **kwargs: "shared_policy",
        )
    )

    # Build algorithm
    algo = ppo.build()

    # Training loop
    logger.info(f"Starting training for {num_iterations} iterations")

    for iteration in range(num_iterations):
        result = algo.train()

        # Log progress
        if iteration % 10 == 0:
            logger.info(f"Iteration {iteration}")
            logger.info(f"  Episode reward mean: {result['episode_reward_mean']:.2f}")
            logger.info(f"  Episode length mean: {result['episode_len_mean']:.2f}")

        # Checkpoint
        if iteration % training_config.get("checkpoint_freq", 10) == 0:
            checkpoint_path = algo.save(checkpoint_dir)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

    # Final checkpoint
    final_checkpoint = algo.save(checkpoint_dir)
    logger.info(f"Training complete. Final checkpoint: {final_checkpoint}")

    # Cleanup
    algo.stop()
    ray.shutdown()

    return final_checkpoint


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train HYPERION swarm policy")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument(
        "--iterations", type=int, default=100, help="Training iterations"
    )
    parser.add_argument(
        "--checkpoint-dir", default="./checkpoints", help="Checkpoint directory"
    )

    args = parser.parse_args()

    train_swarm_policy(
        config_path=args.config,
        num_iterations=args.iterations,
        checkpoint_dir=args.checkpoint_dir,
    )
