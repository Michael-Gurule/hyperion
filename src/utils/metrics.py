"""
Evaluation metrics for HYPERION swarm policies.

Tracks interception rates, fuel efficiency, coordination quality,
and other performance indicators.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""

    episode_id: int
    steps: int
    total_reward: float
    intercepted: bool
    target_escaped: bool
    interception_time: Optional[float] = None
    fuel_remaining: List[float] = field(default_factory=list)
    min_distance_to_target: float = float("inf")
    collision_occurred: bool = False
    communication_failures: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "episode_id": self.episode_id,
            "steps": self.steps,
            "total_reward": self.total_reward,
            "intercepted": self.intercepted,
            "target_escaped": self.target_escaped,
            "interception_time": self.interception_time,
            "avg_fuel_remaining": np.mean(self.fuel_remaining)
            if self.fuel_remaining
            else 0.0,
            "min_fuel_remaining": min(self.fuel_remaining)
            if self.fuel_remaining
            else 0.0,
            "min_distance_to_target": self.min_distance_to_target,
            "collision_occurred": self.collision_occurred,
            "communication_failures": self.communication_failures,
        }


class EvaluationMetrics:
    """
    Tracks and computes evaluation metrics across multiple episodes.
    """

    def __init__(self):
        """Initialize metrics tracker."""
        self.episodes: List[EpisodeMetrics] = []
        self.current_episode: Optional[EpisodeMetrics] = None
        self.episode_count = 0

    def start_episode(self):
        """Start tracking a new episode."""
        self.current_episode = EpisodeMetrics(
            episode_id=self.episode_count,
            steps=0,
            total_reward=0.0,
            intercepted=False,
            target_escaped=False,
        )
        self.episode_count += 1

    def update_step(
        self,
        rewards: Dict[str, float],
        infos: Dict[str, Dict],
        agent_states: Optional[Dict] = None,
        target_state: Optional[Dict] = None,
    ):
        """
        Update metrics for current step.

        Args:
            rewards: Dictionary of agent rewards
            infos: Dictionary of agent info dicts
            agent_states: Optional agent state information
            target_state: Optional target state information
        """
        if self.current_episode is None:
            self.start_episode()

        self.current_episode.steps += 1
        self.current_episode.total_reward += sum(rewards.values())

        # Extract info from first agent (all agents have same global info)
        if len(infos) > 0:
            first_agent_info = list(infos.values())[0]

            if first_agent_info.get("intercepted", False):
                self.current_episode.intercepted = True
                if self.current_episode.interception_time is None:
                    self.current_episode.interception_time = (
                        self.current_episode.steps * 0.1
                    )

            if first_agent_info.get("target_escaped", False):
                self.current_episode.target_escaped = True

        # Track fuel levels
        if agent_states:
            fuel_levels = [state["fuel"] for state in agent_states.values()]
            self.current_episode.fuel_remaining = fuel_levels

        # Track minimum distance to target
        if agent_states and target_state:
            target_pos = target_state["position"]
            for state in agent_states.values():
                distance = np.linalg.norm(state["position"] - target_pos)
                self.current_episode.min_distance_to_target = min(
                    self.current_episode.min_distance_to_target, distance
                )

    def end_episode(self):
        """End current episode and store metrics."""
        if self.current_episode is not None:
            self.episodes.append(self.current_episode)
            self.current_episode = None

    def get_summary_statistics(self) -> Dict:
        """
        Compute summary statistics across all episodes.

        Returns:
            Dictionary of summary metrics
        """
        if len(self.episodes) == 0:
            return {}

        # Interception metrics
        interceptions = [ep.intercepted for ep in self.episodes]
        interception_rate = np.mean(interceptions)

        # Episode length metrics
        episode_lengths = [ep.steps for ep in self.episodes]

        # Reward metrics
        episode_rewards = [ep.total_reward for ep in self.episodes]

        # Fuel efficiency (for successful interceptions)
        successful_episodes = [ep for ep in self.episodes if ep.intercepted]
        if successful_episodes:
            avg_fuel_efficiency = np.mean(
                [np.mean(ep.fuel_remaining) for ep in successful_episodes]
            )
            interception_times = [
                ep.interception_time
                for ep in successful_episodes
                if ep.interception_time is not None
            ]
            avg_interception_time = (
                np.mean(interception_times) if interception_times else None
            )
        else:
            avg_fuel_efficiency = 0.0
            avg_interception_time = None

        # Target escape rate
        escapes = [ep.target_escaped for ep in self.episodes]
        escape_rate = np.mean(escapes)

        # Minimum distance achieved
        min_distances = [ep.min_distance_to_target for ep in self.episodes]

        summary = {
            "num_episodes": len(self.episodes),
            # Success metrics
            "interception_rate": interception_rate,
            "escape_rate": escape_rate,
            "success_rate": interception_rate,  # Alias
            # Episode metrics
            "mean_episode_length": np.mean(episode_lengths),
            "std_episode_length": np.std(episode_lengths),
            "min_episode_length": np.min(episode_lengths),
            "max_episode_length": np.max(episode_lengths),
            # Reward metrics
            "mean_episode_reward": np.mean(episode_rewards),
            "std_episode_reward": np.std(episode_rewards),
            "min_episode_reward": np.min(episode_rewards),
            "max_episode_reward": np.max(episode_rewards),
            # Efficiency metrics
            "mean_fuel_efficiency": avg_fuel_efficiency,
            "mean_interception_time": avg_interception_time,
            # Distance metrics
            "mean_min_distance": np.mean(min_distances),
            "best_min_distance": np.min(min_distances),
        }

        return summary

    def print_summary(self):
        """Print formatted summary statistics."""
        summary = self.get_summary_statistics()

        if not summary:
            print("No episodes evaluated yet.")
            return

        print("\n" + "=" * 60)
        print("HYPERION Evaluation Summary")
        print("=" * 60)

        print(f"\nEpisodes Evaluated: {summary['num_episodes']}")

        print("\n--- Success Metrics ---")
        print(f"Interception Rate: {summary['interception_rate'] * 100:.1f}%")
        print(f"Target Escape Rate: {summary['escape_rate'] * 100:.1f}%")

        print("\n--- Episode Statistics ---")
        print(f"Mean Episode Length: {summary['mean_episode_length']:.1f} steps")
        print(
            f"Episode Length Range: [{summary['min_episode_length']}, {summary['max_episode_length']}]"
        )

        print("\n--- Reward Statistics ---")
        print(f"Mean Episode Reward: {summary['mean_episode_reward']:.2f}")
        print(f"Std Episode Reward: {summary['std_episode_reward']:.2f}")
        print(
            f"Reward Range: [{summary['min_episode_reward']:.2f}, {summary['max_episode_reward']:.2f}]"
        )

        print("\n--- Efficiency Metrics ---")
        print(f"Mean Fuel Efficiency: {summary['mean_fuel_efficiency'] * 100:.1f}%")
        if summary["mean_interception_time"]:
            print(
                f"Mean Interception Time: {summary['mean_interception_time']:.2f} seconds"
            )

        print("\n--- Distance Metrics ---")
        print(f"Mean Min Distance to Target: {summary['mean_min_distance']:.1f} meters")
        print(f"Best Min Distance: {summary['best_min_distance']:.1f} meters")

        print("=" * 60 + "\n")

    def save_results(self, filepath: str):
        """
        Save evaluation results to JSON file.

        Args:
            filepath: Path to save results
        """

        def convert_to_native(obj):
            """Convert numpy types to native Python types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            else:
                return obj

        results = {
            "summary": convert_to_native(self.get_summary_statistics()),
            "episodes": [convert_to_native(ep.to_dict()) for ep in self.episodes],
        }

        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {filepath}")

    def reset(self):
        """Reset all metrics."""
        self.episodes = []
        self.current_episode = None
        self.episode_count = 0


def evaluate_policy(
    env,
    policy,
    num_episodes: int = 100,
    render: bool = False,
    save_path: Optional[str] = None,
) -> EvaluationMetrics:
    """
    Evaluate a trained policy.

    Args:
        env: Environment instance
        policy: Trained policy (should have compute_actions method)
        num_episodes: Number of episodes to evaluate
        render: Whether to render episodes
        save_path: Optional path to save results

    Returns:
        EvaluationMetrics object with results
    """
    metrics = EvaluationMetrics()

    for episode in range(num_episodes):
        metrics.start_episode()

        observations, _ = env.reset()
        done = False

        while not done:
            # Get actions from policy
            if hasattr(policy, "compute_actions"):
                # RLlib policy
                actions = {}
                for agent_id, obs in observations.items():
                    action = policy.compute_single_action(obs)
                    actions[agent_id] = action
            else:
                # Random or custom policy
                actions = {
                    agent: env.action_space(agent).sample() for agent in env.agents
                }

            # Step environment
            observations, rewards, terminations, truncations, infos = env.step(actions)

            # Update metrics
            metrics.update_step(
                rewards=rewards,
                infos=infos,
                agent_states=getattr(env, "agent_states", None),
                target_state=getattr(env, "target_state", None),
            )

            done = all(terminations.values()) or all(truncations.values())

            if render:
                env.render()

        metrics.end_episode()

        if (episode + 1) % 10 == 0:
            print(f"Evaluated {episode + 1}/{num_episodes} episodes...")

    # Print and optionally save results
    metrics.print_summary()

    if save_path:
        metrics.save_results(save_path)

    return metrics
