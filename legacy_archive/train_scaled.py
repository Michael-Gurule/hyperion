"""
Scaled training pipeline for HYPERION.

Integrates:
- GNN-based communication for 50-100+ agents
- MAPPO with centralized critic
- Adversarial training with evasive targets
- Adaptive curriculum learning
"""

import os
import sys
import yaml
import logging
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

import torch
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict
from ray.tune.registry import register_env

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.env.scaled_environment import (
    ScaledHypersonicSwarmEnv,
    create_scaled_env,
    AdversarialConfig,
    RewardConfig,
)
from src.models.gnn_communication import SwarmCoordinationNetwork
from src.training.mappo import MAPPO, MAPPOConfig, MAPPOTrainer
from src.training.curriculum import CurriculumScheduler
from src.utils.logger import setup_logger


logger = setup_logger("hyperion_scaled_training")


@dataclass
class ScaledTrainingConfig:
    """Configuration for scaled training."""

    # Environment
    num_agents: int = 50
    max_steps: int = 500
    arena_size: float = 20000.0

    # Training algorithm
    algorithm: str = "MAPPO"  # "MAPPO", "PPO", "GNN_PPO"

    # MAPPO settings
    use_centralized_critic: bool = True
    share_actor: bool = True

    # GNN settings
    use_gnn: bool = True
    gnn_layers: int = 3
    gnn_hidden_dim: int = 128

    # Training hyperparameters
    total_timesteps: int = 5_000_000
    rollout_length: int = 2048
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    entropy_coeff: float = 0.01
    num_epochs: int = 10
    minibatch_size: int = 256

    # Curriculum settings
    use_curriculum: bool = True
    curriculum_stages: int = 4

    # Adversarial settings
    adversarial_enabled: bool = True
    adversarial_probability: float = 0.3

    # Checkpointing
    checkpoint_freq: int = 50
    checkpoint_dir: str = "./checkpoints/scaled"

    # Logging
    log_interval: int = 10
    eval_interval: int = 50
    eval_episodes: int = 20

    # Hardware
    num_workers: int = 4
    num_gpus: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class GNNPolicyNetwork(TorchModelV2, torch.nn.Module):
    """
    RLlib-compatible GNN policy network.
    Wraps SwarmCoordinationNetwork for use with RLlib.
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config: ModelConfigDict,
        name: str,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        torch.nn.Module.__init__(self)

        custom_config = model_config.get("custom_model_config", {})

        obs_dim = obs_space.shape[0]
        action_dim = action_space.shape[0]

        self.gnn_network = SwarmCoordinationNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=custom_config.get("hidden_dim", 128),
            embed_dim=custom_config.get("embed_dim", 64),
            num_gnn_layers=custom_config.get("gnn_layers", 3),
            num_roles=custom_config.get("num_roles", 4),
            communication_range=custom_config.get("communication_range", 2000.0),
        )

        self._value = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()

        outputs = self.gnn_network(obs)

        # Action output (mean only for RLlib, std is handled separately)
        action_mean = outputs["action_mean"]

        # Store value for value_function()
        self._value = outputs["value"]

        return action_mean, state

    def value_function(self):
        return self._value.squeeze(-1)


class AdaptiveCurriculumScheduler:
    """
    Adaptive curriculum scheduler based on performance metrics.
    """

    def __init__(
        self,
        stages: List[Dict[str, Any]],
        advancement_threshold: float = 0.7,
        regression_threshold: float = 0.3,
        window_size: int = 100,
    ):
        """
        Initialize adaptive curriculum.

        Args:
            stages: List of curriculum stage configurations
            advancement_threshold: Success rate to advance
            regression_threshold: Success rate to regress
            window_size: Episodes for moving average
        """
        self.stages = stages
        self.current_stage = 0
        self.advancement_threshold = advancement_threshold
        self.regression_threshold = regression_threshold
        self.window_size = window_size

        self.episode_results = []
        self.stage_history = []

    def get_current_config(self) -> Dict[str, Any]:
        """Get current stage configuration."""
        return self.stages[self.current_stage]

    def update(self, success: bool, episode_reward: float) -> bool:
        """
        Update curriculum based on episode result.

        Args:
            success: Whether episode was successful (interception)
            episode_reward: Episode reward

        Returns:
            stage_changed: Whether stage changed
        """
        self.episode_results.append({
            "success": success,
            "reward": episode_reward,
            "stage": self.current_stage,
        })

        # Compute moving average success rate
        recent = self.episode_results[-self.window_size:]
        success_rate = np.mean([r["success"] for r in recent])

        stage_changed = False

        # Check for advancement
        if (
            success_rate >= self.advancement_threshold
            and self.current_stage < len(self.stages) - 1
        ):
            self.current_stage += 1
            stage_changed = True
            logger.info(
                f"Curriculum advanced to stage {self.current_stage}: "
                f"{self.stages[self.current_stage]['name']}"
            )

        # Check for regression (optional)
        elif (
            success_rate <= self.regression_threshold
            and self.current_stage > 0
            and len(recent) >= self.window_size
        ):
            self.current_stage -= 1
            stage_changed = True
            logger.info(
                f"Curriculum regressed to stage {self.current_stage}: "
                f"{self.stages[self.current_stage]['name']}"
            )

        if stage_changed:
            self.stage_history.append({
                "episode": len(self.episode_results),
                "new_stage": self.current_stage,
                "success_rate": success_rate,
            })

        return stage_changed


def create_curriculum_stages(config: ScaledTrainingConfig) -> List[Dict[str, Any]]:
    """Create curriculum stages for scaled training."""
    stages = [
        {
            "name": "basic",
            "num_agents": min(10, config.num_agents // 5),
            "target_speed": 500.0,
            "adversarial_enabled": False,
            "arena_size": config.arena_size * 0.5,
        },
        {
            "name": "intermediate",
            "num_agents": min(25, config.num_agents // 2),
            "target_speed": 1000.0,
            "adversarial_enabled": False,
            "arena_size": config.arena_size * 0.75,
        },
        {
            "name": "advanced",
            "num_agents": config.num_agents,
            "target_speed": 1500.0,
            "adversarial_enabled": True,
            "arena_size": config.arena_size,
        },
        {
            "name": "expert",
            "num_agents": config.num_agents,
            "target_speed": 1700.0,
            "adversarial_enabled": True,
            "arena_size": config.arena_size,
            "adversarial_probability": 0.5,
        },
    ]

    return stages[:config.curriculum_stages]


def train_with_mappo(config: ScaledTrainingConfig) -> Dict[str, Any]:
    """
    Train using MAPPO algorithm.

    Args:
        config: Training configuration

    Returns:
        Training results
    """
    logger.info("Starting MAPPO training")
    logger.info(f"Config: {config}")

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Setup curriculum
    curriculum_stages = create_curriculum_stages(config)
    curriculum = AdaptiveCurriculumScheduler(
        stages=curriculum_stages,
        advancement_threshold=0.7,
    ) if config.use_curriculum else None

    # Get initial curriculum config
    stage_config = curriculum.get_current_config() if curriculum else {}

    # Create environment
    env_config = {
        "num_agents": stage_config.get("num_agents", config.num_agents),
        "max_steps": config.max_steps,
        "arena_size": stage_config.get("arena_size", config.arena_size),
        "target_speed": stage_config.get("target_speed", 1700.0),
    }

    adversarial_config = AdversarialConfig(
        enabled=stage_config.get("adversarial_enabled", config.adversarial_enabled),
        evasion_probability=stage_config.get(
            "adversarial_probability", config.adversarial_probability
        ),
    )

    env = create_scaled_env(
        num_agents=env_config["num_agents"],
        adversarial=adversarial_config.enabled,
        max_steps=env_config["max_steps"],
        arena_size=env_config["arena_size"],
        target_speed=env_config["target_speed"],
    )

    # Create MAPPO config
    mappo_config = MAPPOConfig(
        actor_lr=config.learning_rate,
        critic_lr=config.learning_rate * 3,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_param=config.clip_param,
        entropy_coeff=config.entropy_coeff,
        num_epochs=config.num_epochs,
        hidden_dim=config.gnn_hidden_dim,
        use_centralized_critic=config.use_centralized_critic,
        share_actor=config.share_actor,
        use_gnn=config.use_gnn,
        gnn_layers=config.gnn_layers,
        device=config.device,
    )

    # Create trainer
    trainer = MAPPOTrainer(env=env, config=mappo_config)

    # Training metrics
    history = defaultdict(list)
    best_reward = float("-inf")

    timesteps = 0
    episodes = 0

    while timesteps < config.total_timesteps:
        # Collect rollout
        observations, _ = env.reset()
        episode_reward = 0.0
        episode_success = False
        steps = 0

        for _ in range(config.rollout_length):
            actions, log_probs, values = trainer.mappo.select_actions(observations)
            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            dones = {
                aid: terminations.get(aid, False) or truncations.get(aid, False)
                for aid in observations.keys()
            }

            trainer.mappo.store_transition(
                observations, actions, rewards, dones, log_probs, values
            )

            episode_reward += np.mean(list(rewards.values()))
            steps += 1
            timesteps += len(observations)

            if all(dones.values()) or len(next_obs) == 0:
                episodes += 1

                # Check success
                episode_success = any(
                    info.get("intercepted", False) for info in infos.values()
                )

                # Update curriculum
                if curriculum:
                    stage_changed = curriculum.update(episode_success, episode_reward)
                    if stage_changed:
                        # Recreate environment with new config
                        stage_config = curriculum.get_current_config()
                        env = create_scaled_env(
                            num_agents=stage_config.get("num_agents", config.num_agents),
                            adversarial=stage_config.get("adversarial_enabled", True),
                            max_steps=config.max_steps,
                            arena_size=stage_config.get("arena_size", config.arena_size),
                            target_speed=stage_config.get("target_speed", 1700.0),
                        )

                # Log
                history["episode_reward"].append(episode_reward)
                history["episode_length"].append(steps)
                history["success"].append(float(episode_success))

                if episodes % config.log_interval == 0:
                    mean_reward = np.mean(history["episode_reward"][-config.log_interval:])
                    mean_success = np.mean(history["success"][-config.log_interval:])
                    logger.info(
                        f"Episode {episodes} | Steps {timesteps:,} | "
                        f"Reward: {mean_reward:.2f} | Success: {mean_success:.1%}"
                    )

                # Reset
                observations, _ = env.reset()
                episode_reward = 0.0
                steps = 0
            else:
                observations = next_obs

        # Update policy
        _, _, last_values = trainer.mappo.select_actions(observations)
        update_stats = trainer.mappo.update(last_values)

        history["policy_loss"].append(update_stats["policy_loss"])
        history["value_loss"].append(update_stats["value_loss"])
        history["entropy"].append(update_stats["entropy"])

        # Checkpoint
        if episodes % config.checkpoint_freq == 0:
            checkpoint_path = os.path.join(
                config.checkpoint_dir, f"mappo_ep{episodes}.pt"
            )
            trainer.mappo.save(checkpoint_path)

            mean_reward = np.mean(history["episode_reward"][-100:])
            if mean_reward > best_reward:
                best_reward = mean_reward
                best_path = os.path.join(config.checkpoint_dir, "mappo_best.pt")
                trainer.mappo.save(best_path)
                logger.info(f"New best model saved: {mean_reward:.2f}")

        # Evaluation
        if episodes % config.eval_interval == 0:
            eval_results = trainer.evaluate(
                num_episodes=config.eval_episodes,
                deterministic=True,
            )
            logger.info(
                f"Evaluation: Reward={eval_results['mean_reward']:.2f}, "
                f"Success={eval_results['success_rate']:.1%}"
            )
            history["eval_reward"].append(eval_results["mean_reward"])
            history["eval_success"].append(eval_results["success_rate"])

    # Final save
    final_path = os.path.join(config.checkpoint_dir, "mappo_final.pt")
    trainer.mappo.save(final_path)

    return dict(history)


def train_with_rllib(config: ScaledTrainingConfig) -> Dict[str, Any]:
    """
    Train using RLlib with optional GNN policy.

    Args:
        config: Training configuration

    Returns:
        Training results
    """
    logger.info("Starting RLlib training")

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Register custom model
    if config.use_gnn:
        ModelCatalog.register_custom_model("gnn_policy", GNNPolicyNetwork)

    # Register environment
    def env_creator(env_config):
        return create_scaled_env(
            num_agents=env_config.get("num_agents", config.num_agents),
            adversarial=env_config.get("adversarial_enabled", config.adversarial_enabled),
            max_steps=env_config.get("max_steps", config.max_steps),
            arena_size=env_config.get("arena_size", config.arena_size),
            target_speed=env_config.get("target_speed", 1700.0),
        )

    register_env("hyperion_scaled", env_creator)

    # Create environment to get spaces
    env_config = {
        "num_agents": config.num_agents,
        "max_steps": config.max_steps,
        "arena_size": config.arena_size,
    }
    dummy_env = env_creator(env_config)
    agent_ids = dummy_env.possible_agents
    obs_space = dummy_env.observation_space(agent_ids[0])
    act_space = dummy_env.action_space(agent_ids[0])

    # Build PPO config
    model_config = {}
    if config.use_gnn:
        model_config = {
            "custom_model": "gnn_policy",
            "custom_model_config": {
                "hidden_dim": config.gnn_hidden_dim,
                "embed_dim": 64,
                "gnn_layers": config.gnn_layers,
                "num_roles": 4,
                "communication_range": 2000.0,
            },
        }

    ppo_config = (
        PPOConfig()
        .environment("hyperion_scaled", env_config=env_config)
        .framework("torch")
        .resources(
            num_gpus=config.num_gpus,
            num_cpus_per_worker=1,
        )
        .env_runners(
            num_env_runners=config.num_workers,
            rollout_fragment_length=config.rollout_length // config.num_workers,
        )
        .training(
            lr=config.learning_rate,
            gamma=config.gamma,
            lambda_=config.gae_lambda,
            clip_param=config.clip_param,
            entropy_coeff=config.entropy_coeff,
            num_epochs=config.num_epochs,
            minibatch_size=config.minibatch_size,
            train_batch_size=config.rollout_length,
            model=model_config if config.use_gnn else {},
        )
        .multi_agent(
            policies={"shared_policy": (None, obs_space, act_space, {})},
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "shared_policy",
        )
    )

    # Build and train
    algo = ppo_config.build()

    history = defaultdict(list)
    best_reward = float("-inf")

    for iteration in range(config.total_timesteps // config.rollout_length):
        result = algo.train()

        history["episode_reward"].append(result.get("episode_reward_mean", 0))
        history["episode_length"].append(result.get("episode_len_mean", 0))

        if iteration % config.log_interval == 0:
            logger.info(
                f"Iteration {iteration} | "
                f"Reward: {result.get('episode_reward_mean', 0):.2f} | "
                f"Length: {result.get('episode_len_mean', 0):.1f}"
            )

        if iteration % config.checkpoint_freq == 0:
            checkpoint = algo.save(config.checkpoint_dir)
            logger.info(f"Checkpoint: {checkpoint}")

            if result.get("episode_reward_mean", 0) > best_reward:
                best_reward = result["episode_reward_mean"]

    algo.stop()
    ray.shutdown()

    return dict(history)


def main():
    """Main training entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Train scaled HYPERION")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Config file"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="MAPPO",
        choices=["MAPPO", "PPO", "GNN_PPO"],
        help="Training algorithm",
    )
    parser.add_argument(
        "--num-agents", type=int, default=50, help="Number of agents"
    )
    parser.add_argument(
        "--timesteps", type=int, default=5_000_000, help="Total timesteps"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints/scaled",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--no-curriculum", action="store_true", help="Disable curriculum"
    )
    parser.add_argument(
        "--no-adversarial", action="store_true", help="Disable adversarial"
    )

    args = parser.parse_args()

    # Load config file if exists
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            file_config = yaml.safe_load(f)
    else:
        file_config = {}

    # Build training config
    config = ScaledTrainingConfig(
        algorithm=args.algorithm,
        num_agents=args.num_agents,
        total_timesteps=args.timesteps,
        checkpoint_dir=args.checkpoint_dir,
        use_curriculum=not args.no_curriculum,
        adversarial_enabled=not args.no_adversarial,
        use_gnn=args.algorithm in ["GNN_PPO", "MAPPO"],
    )

    # Run training
    if args.algorithm == "MAPPO":
        history = train_with_mappo(config)
    else:
        history = train_with_rllib(config)

    # Save training history
    history_path = os.path.join(config.checkpoint_dir, "training_history.yaml")
    with open(history_path, "w") as f:
        yaml.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f)

    logger.info(f"Training complete. History saved to {history_path}")


if __name__ == "__main__":
    main()
