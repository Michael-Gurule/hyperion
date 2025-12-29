"""
Enhanced training pipeline for HYPERION with all new components.

Integrates:
- Projectile system with PN guidance
- Parameterized curriculum learning (speed/evasion progression)
- Intrinsic rewards (anti-trailing, geometry bonus, novelty)
- Hierarchical role-based policies (Options Framework)
"""

import os
import sys
import yaml
import logging
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

import torch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.env.scaled_environment import (
    ScaledHypersonicSwarmEnv,
    AdversarialConfig,
    RewardConfig,
)
from src.training.curriculum import (
    ParameterizedCurriculumScheduler,
    ParameterizedCurriculumConfig,
    EvasionLevel,
    create_default_curriculum,
)
from src.training.intrinsic_rewards import (
    IntrinsicRewardConfig,
    IntrinsicRewardCalculator,
)
from src.models.hierarchical_policy import (
    HierarchicalPolicyConfig,
    HierarchicalMAPPO,
    RoleType,
)
from src.utils.logger import setup_logger


logger = setup_logger("hyperion_enhanced_training")


@dataclass
class EnhancedTrainingConfig:
    """Configuration for enhanced training with all new components."""

    # Environment
    num_agents: int = 50
    max_steps: int = 500
    arena_size: float = 8000.0  # Reduced from 20000 per plan
    initial_spread_min: float = 2000.0
    initial_spread_max: float = 4000.0

    # Projectile settings
    use_projectiles: bool = True
    projectile_speed: float = 600.0
    projectile_cooldown: float = 2.0

    # Curriculum settings
    use_curriculum: bool = True
    agent_max_speed: float = 300.0
    advancement_threshold: float = 0.7
    regression_threshold: float = 0.3

    # Hierarchical policy settings
    use_hierarchical: bool = True
    role_update_interval: int = 10
    manager_lr_scale: float = 0.5

    # Intrinsic rewards
    use_intrinsic_rewards: bool = True
    trailing_penalty_scale: float = 0.5
    geometry_bonus_scale: float = 1.0
    novelty_bonus_scale: float = 0.1

    # Training hyperparameters
    total_timesteps: int = 2_000_000
    rollout_length: int = 1024
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    entropy_coeff: float = 0.01
    num_epochs: int = 10

    # Checkpointing
    checkpoint_freq: int = 50
    checkpoint_dir: str = "./checkpoints/enhanced"

    # Logging
    log_interval: int = 10
    eval_interval: int = 50
    eval_episodes: int = 10

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def create_enhanced_env(
    config: EnhancedTrainingConfig,
    curriculum: Optional[ParameterizedCurriculumScheduler] = None,
) -> ScaledHypersonicSwarmEnv:
    """Create environment with curriculum settings applied."""

    # Get curriculum config if available
    if curriculum:
        env_config = curriculum.get_env_config()
        adv_config = curriculum.get_adversarial_config()
    else:
        env_config = {
            "target_speed": 450.0,  # Default: 1.5x agent speed
            "evasion_probability": 0.0,
            "target_behavior": "BALLISTIC",
        }
        adv_config = {"enabled": False}

    # Create adversarial config
    adversarial = AdversarialConfig(
        enabled=adv_config.get("enabled", False),
        evasion_probability=adv_config.get("evasion_probability", 0.0),
        evasive_maneuvers=adv_config.get("evasive_maneuvers", False),
        jink_frequency=adv_config.get("jink_frequency", 0.0),
        jink_magnitude=adv_config.get("jink_magnitude", 0.0),
    )

    # Create reward config with projectile rewards
    rewards = RewardConfig(
        intercept_reward=100.0,
        distance_scale=0.1,
        fuel_penalty=0.01,
        formation_bonus=0.5,
        projectile_hit_bonus=50.0 if config.use_projectiles else 0.0,
        projectile_launch_cost=-0.5 if config.use_projectiles else 0.0,
    )

    # Create environment directly with full config control
    return ScaledHypersonicSwarmEnv(
        num_agents=config.num_agents,
        max_steps=config.max_steps,
        arena_size=config.arena_size,
        target_speed=env_config["target_speed"],
        adversarial_config=adversarial,
        reward_config=rewards,
        use_projectiles=config.use_projectiles,
    )


def train_enhanced(config: EnhancedTrainingConfig) -> Dict[str, Any]:
    """
    Train with all enhanced components.

    Args:
        config: Training configuration

    Returns:
        Training history
    """
    logger.info("=" * 60)
    logger.info("HYPERION Enhanced Training")
    logger.info("=" * 60)
    logger.info(f"Device: {config.device}")
    logger.info(f"Projectiles: {config.use_projectiles}")
    logger.info(f"Hierarchical policies: {config.use_hierarchical}")
    logger.info(f"Intrinsic rewards: {config.use_intrinsic_rewards}")
    logger.info(f"Curriculum: {config.use_curriculum}")
    logger.info("=" * 60)

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Setup curriculum
    curriculum = None
    if config.use_curriculum:
        curriculum_config = ParameterizedCurriculumConfig(
            agent_max_speed=config.agent_max_speed,
            advancement_threshold=config.advancement_threshold,
            regression_threshold=config.regression_threshold,
        )
        curriculum = ParameterizedCurriculumScheduler(config=curriculum_config)
        logger.info(f"Curriculum initialized with {curriculum.num_stages} stages")
        logger.info(f"Starting stage: {curriculum.stage_name}")

    # Create environment
    env = create_enhanced_env(config, curriculum)
    logger.info(f"Environment created with {config.num_agents} agents")

    # Get observation and action dimensions
    sample_obs = env.reset()[0]
    sample_agent = list(sample_obs.keys())[0]
    obs_dim = sample_obs[sample_agent].shape[0]
    action_dim = 3 if config.use_projectiles else 2

    logger.info(f"Observation dim: {obs_dim}, Action dim: {action_dim}")

    # Setup intrinsic rewards
    intrinsic_calculator = None
    if config.use_intrinsic_rewards:
        intrinsic_config = IntrinsicRewardConfig(
            trailing_penalty_scale=config.trailing_penalty_scale,
            geometry_bonus_scale=config.geometry_bonus_scale,
            novelty_bonus_scale=config.novelty_bonus_scale,
        )
        intrinsic_calculator = IntrinsicRewardCalculator(config=intrinsic_config)
        logger.info("Intrinsic reward calculator initialized")

    # Setup policy
    if config.use_hierarchical:
        policy_config = HierarchicalPolicyConfig(
            action_dim=action_dim,
            role_update_interval=config.role_update_interval,
            manager_lr_scale=config.manager_lr_scale,
        )
        trainer = HierarchicalMAPPO(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_agents=config.num_agents,
            config=policy_config,
            device=config.device,
        )
        logger.info("Hierarchical MAPPO initialized with role-based policies")
    else:
        # Fall back to standard MAPPO
        from src.training.mappo import MAPPO, MAPPOConfig

        mappo_config = MAPPOConfig(
            actor_lr=config.learning_rate,
            critic_lr=config.learning_rate * 3,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_param=config.clip_param,
            entropy_coeff=config.entropy_coeff,
            num_epochs=config.num_epochs,
            device=config.device,
        )
        trainer = MAPPO(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_agents=config.num_agents,
            config=mappo_config,
        )
        logger.info("Standard MAPPO initialized")

    # Training metrics
    history = defaultdict(list)
    best_reward = float("-inf")
    best_success_rate = 0.0

    timesteps = 0
    episodes = 0
    total_successes = 0

    logger.info("\nStarting training loop...")
    logger.info("-" * 60)

    while timesteps < config.total_timesteps:
        # Collect rollout
        observations, _ = env.reset()

        if intrinsic_calculator:
            intrinsic_calculator.reset()

        episode_reward = 0.0
        episode_intrinsic_reward = 0.0
        episode_success = False
        steps = 0
        role_distribution = defaultdict(int)

        for step in range(config.rollout_length):
            # Select actions
            if config.use_hierarchical:
                actions, log_probs, values, roles = trainer.select_actions(observations)
                # Track role distribution
                for aid, role in roles.items():
                    role_distribution[RoleType(role).name] += 1
            else:
                actions, log_probs, values = trainer.select_actions(observations)
                roles = {aid: 0 for aid in observations}

            # Step environment
            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            # Compute intrinsic rewards
            intrinsic_rewards = {}
            if intrinsic_calculator and len(next_obs) > 0 and env.target_state is not None:
                # Get target info from environment
                target_pos = env.target_state.get("position", np.zeros(2))
                target_vel = env.target_state.get("velocity", np.zeros(2))

                for aid in observations:
                    if aid in next_obs:
                        agent_obs = next_obs[aid]
                        # Extract position and velocity from observation
                        agent_pos = agent_obs[:2]
                        agent_vel = agent_obs[2:4] if len(agent_obs) > 3 else np.zeros(2)

                        intrinsic, _ = intrinsic_calculator.compute_agent_reward(
                            agent_id=aid,
                            agent_position=agent_pos,
                            agent_velocity=agent_vel,
                            target_position=target_pos,
                            target_velocity=target_vel,
                            detection_range=2000.0,
                            include_novelty=True,
                        )
                        intrinsic_rewards[aid] = intrinsic
                        episode_intrinsic_reward += intrinsic

            # Combine rewards
            combined_rewards = {}
            for aid in rewards:
                combined_rewards[aid] = rewards[aid] + intrinsic_rewards.get(aid, 0.0)

            dones = {
                aid: terminations.get(aid, False) or truncations.get(aid, False)
                for aid in observations.keys()
            }

            # Store transition
            if config.use_hierarchical:
                trainer.store_transition(
                    observations, actions, combined_rewards, dones, log_probs, values, roles
                )
            else:
                trainer.store_transition(
                    observations, actions, combined_rewards, dones, log_probs, values
                )

            episode_reward += np.mean(list(rewards.values()))
            steps += 1
            timesteps += len(observations)

            # Check episode end
            if all(dones.values()) or len(next_obs) == 0:
                episodes += 1

                # Check success
                episode_success = any(
                    info.get("intercepted", False) for info in infos.values()
                )
                if episode_success:
                    total_successes += 1

                # Update curriculum
                if curriculum:
                    result = curriculum.update(episode_success, episode_reward)
                    if result["stage_changed"]:
                        logger.info(
                            f"\n{'='*40}\n"
                            f"Curriculum stage change: {result['direction']}\n"
                            f"New stage: {result.get('new_stage_name', 'unknown')}\n"
                            f"Success rate: {result['success_rate']:.1%}\n"
                            f"{'='*40}\n"
                        )
                        # Recreate environment with new settings
                        env = create_enhanced_env(config, curriculum)

                # Log metrics
                history["episode_reward"].append(episode_reward)
                history["episode_length"].append(steps)
                history["success"].append(float(episode_success))
                history["intrinsic_reward"].append(episode_intrinsic_reward)

                if curriculum:
                    history["curriculum_stage"].append(curriculum.current_stage_idx)

                if episodes % config.log_interval == 0:
                    recent_rewards = history["episode_reward"][-config.log_interval:]
                    recent_success = history["success"][-config.log_interval:]

                    mean_reward = np.mean(recent_rewards)
                    mean_success = np.mean(recent_success)
                    overall_success = total_successes / episodes

                    stage_info = ""
                    if curriculum:
                        stage_info = f" | Stage: {curriculum.current_stage_idx + 1}/{curriculum.num_stages}"

                    role_info = ""
                    if config.use_hierarchical and role_distribution:
                        total_roles = sum(role_distribution.values())
                        role_pcts = {k: v / total_roles * 100 for k, v in role_distribution.items()}
                        role_info = f"\n  Roles: " + ", ".join(f"{k}:{v:.0f}%" for k, v in role_pcts.items())

                    logger.info(
                        f"Ep {episodes:5d} | Steps {timesteps:8,d} | "
                        f"Reward: {mean_reward:7.2f} | "
                        f"Success: {mean_success:5.1%} (overall: {overall_success:5.1%})"
                        f"{stage_info}{role_info}"
                    )

                # Reset for new episode
                observations, _ = env.reset()
                if intrinsic_calculator:
                    intrinsic_calculator.reset()
                episode_reward = 0.0
                episode_intrinsic_reward = 0.0
                steps = 0
                role_distribution = defaultdict(int)
            else:
                observations = next_obs

        # Update policy
        if config.use_hierarchical:
            _, _, last_values, _ = trainer.select_actions(observations)
        else:
            _, _, last_values = trainer.select_actions(observations)

        update_stats = trainer.update(
            last_values,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_param=config.clip_param,
            num_epochs=config.num_epochs,
        )

        history["policy_loss"].append(update_stats.get("policy_loss", 0))
        history["value_loss"].append(update_stats.get("value_loss", 0))
        history["entropy"].append(update_stats.get("entropy", 0))

        if config.use_hierarchical:
            history["role_entropy"].append(update_stats.get("role_entropy", 0))

        # Checkpoint
        if episodes > 0 and episodes % config.checkpoint_freq == 0:
            checkpoint_path = os.path.join(
                config.checkpoint_dir, f"enhanced_ep{episodes}.pt"
            )
            trainer.save(checkpoint_path)

            # Save best model
            recent_rewards = history["episode_reward"][-100:] if len(history["episode_reward"]) >= 100 else history["episode_reward"]
            recent_success = history["success"][-100:] if len(history["success"]) >= 100 else history["success"]

            mean_reward = np.mean(recent_rewards)
            mean_success = np.mean(recent_success)

            if mean_success > best_success_rate or (mean_success == best_success_rate and mean_reward > best_reward):
                best_success_rate = mean_success
                best_reward = mean_reward
                best_path = os.path.join(config.checkpoint_dir, "enhanced_best.pt")
                trainer.save(best_path)
                logger.info(f"New best model: success={mean_success:.1%}, reward={mean_reward:.2f}")

    # Final save
    final_path = os.path.join(config.checkpoint_dir, "enhanced_final.pt")
    trainer.save(final_path)

    # Save training history
    history_path = os.path.join(config.checkpoint_dir, "training_history.yaml")
    with open(history_path, "w") as f:
        yaml.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f)

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Total episodes: {episodes}")
    logger.info(f"Total timesteps: {timesteps:,}")
    logger.info(f"Final success rate: {total_successes / episodes:.1%}")
    logger.info(f"Best success rate: {best_success_rate:.1%}")
    if curriculum:
        logger.info(f"Final curriculum stage: {curriculum.stage_name}")
        logger.info(f"Stage transitions: {len(curriculum.stage_history)}")
    logger.info(f"Checkpoints saved to: {config.checkpoint_dir}")
    logger.info("=" * 60)

    return dict(history)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced HYPERION Training")
    parser.add_argument("--timesteps", type=int, default=2_000_000, help="Total timesteps")
    parser.add_argument("--num-agents", type=int, default=50, help="Number of agents")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/enhanced")
    parser.add_argument("--no-projectiles", action="store_true", help="Disable projectiles")
    parser.add_argument("--no-hierarchical", action="store_true", help="Disable hierarchical policies")
    parser.add_argument("--no-intrinsic", action="store_true", help="Disable intrinsic rewards")
    parser.add_argument("--no-curriculum", action="store_true", help="Disable curriculum")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")

    args = parser.parse_args()

    # Build config
    config = EnhancedTrainingConfig(
        total_timesteps=args.timesteps,
        num_agents=args.num_agents,
        checkpoint_dir=args.checkpoint_dir,
        use_projectiles=not args.no_projectiles,
        use_hierarchical=not args.no_hierarchical,
        use_intrinsic_rewards=not args.no_intrinsic,
        use_curriculum=not args.no_curriculum,
    )

    if args.device:
        config.device = args.device

    # Run training
    history = train_enhanced(config)

    return history


if __name__ == "__main__":
    main()
