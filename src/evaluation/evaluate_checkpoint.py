"""
Evaluate a trained checkpoint.

Loads a saved model checkpoint and runs evaluation episodes,
outputting metrics to JSON.

Usage:
    python -m src.evaluation.evaluate_checkpoint \
        --checkpoint ./checkpoints/full_curriculum/enhanced_best.pt \
        --num-episodes 100 \
        --output ./outputs/eval_results.json
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.evaluation.metrics import EvaluationMetrics
from src.env.scaled_environment import ScaledHypersonicSwarmEnv
from src.models.hierarchical_policy import HierarchicalMAPPO, HierarchicalPolicyConfig
from src.training.mappo import MAPPO, MAPPOConfig
from src.utils.logger import setup_logger


logger = setup_logger("evaluate_checkpoint")


def detect_checkpoint_type(checkpoint_path: str) -> str:
    """
    Detect whether checkpoint is from HierarchicalMAPPO or MAPPO.

    Args:
        checkpoint_path: Path to .pt checkpoint file

    Returns:
        "hierarchical" or "mappo"
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "policy_state_dict" in checkpoint:
        return "hierarchical"
    elif "actors" in checkpoint:
        return "mappo"
    else:
        raise ValueError(f"Unknown checkpoint format. Keys: {list(checkpoint.keys())}")


def load_trainer(
    checkpoint_path: str,
    obs_dim: int,
    action_dim: int,
    num_agents: int,
    device: str,
):
    """
    Load the appropriate trainer from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint
        obs_dim: Observation dimension
        action_dim: Action dimension
        num_agents: Number of agents
        device: Device to load model on

    Returns:
        Loaded trainer (HierarchicalMAPPO or MAPPO)
    """
    checkpoint_type = detect_checkpoint_type(checkpoint_path)
    logger.info(f"Detected checkpoint type: {checkpoint_type}")

    if checkpoint_type == "hierarchical":
        config = HierarchicalPolicyConfig(action_dim=action_dim)
        trainer = HierarchicalMAPPO(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_agents=num_agents,
            config=config,
            device=device,
        )
    else:
        config = MAPPOConfig(device=device)
        trainer = MAPPO(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_agents=num_agents,
            config=config,
        )

    trainer.load(checkpoint_path)
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    return trainer, checkpoint_type


def evaluate_checkpoint(
    checkpoint_path: str,
    num_episodes: int = 100,
    num_agents: int = 50,
    use_projectiles: bool = True,
    deterministic: bool = True,
    device: str = None,
    output_path: str = None,
) -> EvaluationMetrics:
    """
    Evaluate a trained checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        num_episodes: Number of evaluation episodes
        num_agents: Number of agents in environment
        use_projectiles: Whether environment uses projectiles
        deterministic: Use deterministic (mean) actions
        device: Device to run on (None for auto-detect)
        output_path: Path to save results JSON

    Returns:
        EvaluationMetrics with results
    """
    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    logger.info(f"Using device: {device}")

    # Create environment
    env = ScaledHypersonicSwarmEnv(
        num_agents=num_agents,
        max_steps=500,
        use_projectiles=use_projectiles,
    )
    logger.info(f"Created environment with {num_agents} agents")

    # Get dimensions from environment
    sample_obs, _ = env.reset()
    sample_agent = list(sample_obs.keys())[0]
    obs_dim = sample_obs[sample_agent].shape[0]
    action_dim = 3 if use_projectiles else 2

    logger.info(f"Observation dim: {obs_dim}, Action dim: {action_dim}")

    # Load trainer
    trainer, checkpoint_type = load_trainer(
        checkpoint_path=checkpoint_path,
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_agents=num_agents,
        device=device,
    )

    # Initialize metrics
    metrics = EvaluationMetrics()

    logger.info(f"Starting evaluation for {num_episodes} episodes...")
    logger.info("-" * 60)

    for episode in range(num_episodes):
        metrics.start_episode()
        observations, _ = env.reset()
        done = False

        while not done:
            # Get actions from trained policy
            if checkpoint_type == "hierarchical":
                actions, _, _, _ = trainer.select_actions(
                    observations, deterministic=deterministic
                )
            else:
                actions, _, _ = trainer.select_actions(
                    observations, deterministic=deterministic
                )

            # Step environment
            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            # Update metrics
            metrics.update_step(
                rewards=rewards,
                infos=infos,
                agent_states=getattr(env, "agent_states", None),
                target_state=getattr(env, "target_state", None),
            )

            # Check if done
            done = all(terminations.values()) or all(truncations.values()) or len(next_obs) == 0
            observations = next_obs

        metrics.end_episode()

        # Progress logging
        if (episode + 1) % 10 == 0:
            recent = metrics.episodes[-10:]
            recent_success = np.mean([ep.intercepted for ep in recent])
            recent_reward = np.mean([ep.total_reward for ep in recent])
            logger.info(
                f"Episode {episode + 1}/{num_episodes} | "
                f"Recent success: {recent_success:.1%} | "
                f"Recent reward: {recent_reward:.1f}"
            )

    # Print summary
    metrics.print_summary()

    # Save results
    if output_path:
        metrics.save_results(output_path)
        logger.info(f"Results saved to {output_path}")

    return metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained HYPERION checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to .pt checkpoint file",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes (default: 100)",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=50,
        help="Number of agents (default: 50)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/eval_results.json",
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--no-projectiles",
        action="store_true",
        help="Disable projectiles in environment",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions instead of deterministic",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "mps", "cuda"],
        help="Device to run on (default: auto-detect)",
    )

    args = parser.parse_args()

    # Validate checkpoint exists
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Create output directory
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Run evaluation
    metrics = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        num_episodes=args.num_episodes,
        num_agents=args.num_agents,
        use_projectiles=not args.no_projectiles,
        deterministic=not args.stochastic,
        device=args.device,
        output_path=args.output,
    )

    return metrics


if __name__ == "__main__":
    main()
