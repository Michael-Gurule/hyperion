"""
Intrinsic reward mechanisms for enhanced exploration.

Includes:
- Velocity mismatch penalty (anti-trailing)
- Novelty search via state embedding archive
- Intercept geometry rewards
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
from collections import deque


@dataclass
class IntrinsicRewardConfig:
    """Configuration for intrinsic reward computation."""

    # Anti-trailing penalties
    trailing_penalty_scale: float = 0.5
    trailing_threshold: float = 0.7  # Cosine similarity above this = trailing
    speed_ratio_threshold: float = 0.8  # Penalize more if slower than target

    # Intercept geometry
    geometry_bonus_scale: float = 1.0
    good_approach_angle: float = np.pi / 4  # 45 degrees from behind

    # Novelty search
    novelty_bonus_scale: float = 0.1
    novelty_k_neighbors: int = 10
    archive_max_size: int = 10000
    embedding_dim: int = 16

    # Curiosity/exploration
    detection_bonus: float = 0.3
    new_detection_bonus: float = 1.0
    coverage_bonus_scale: float = 0.2


class VelocityMismatchPenalty:
    """
    Computes anti-trailing penalty based on velocity alignment.

    Penalizes agents that:
    - Move in same direction as target (velocity aligned)
    - Are slower than target while trailing
    - Are moving away from target
    """

    def __init__(self, config: Optional[IntrinsicRewardConfig] = None):
        """Initialize velocity mismatch penalty calculator."""
        self.config = config or IntrinsicRewardConfig()

    def compute(
        self,
        agent_velocity: np.ndarray,
        target_velocity: np.ndarray,
        agent_position: np.ndarray,
        target_position: np.ndarray,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute anti-trailing penalty.

        Args:
            agent_velocity: Agent velocity vector [vx, vy]
            target_velocity: Target velocity vector [vx, vy]
            agent_position: Agent position [x, y]
            target_position: Target position [x, y]

        Returns:
            penalty: Negative reward (penalty) value
            info: Debug info dict with component breakdowns
        """
        agent_speed = np.linalg.norm(agent_velocity)
        target_speed = np.linalg.norm(target_velocity)

        info = {
            "agent_speed": agent_speed,
            "target_speed": target_speed,
            "trailing_score": 0.0,
            "speed_penalty": 0.0,
            "receding_penalty": 0.0,
            "total_penalty": 0.0,
        }

        total_penalty = 0.0

        # Skip if either entity is stationary
        if agent_speed < 1e-6 or target_speed < 1e-6:
            return 0.0, info

        # Normalize velocities
        agent_dir = agent_velocity / agent_speed
        target_dir = target_velocity / target_speed

        # 1. Trailing penalty: penalize moving in same direction as target
        trailing_score = np.dot(agent_dir, target_dir)
        info["trailing_score"] = trailing_score

        if trailing_score > self.config.trailing_threshold:
            # Moving same direction as target (trailing)
            trailing_penalty = self.config.trailing_penalty_scale * trailing_score

            # 2. Speed penalty: penalize more if slower than target
            speed_ratio = agent_speed / target_speed
            if speed_ratio < self.config.speed_ratio_threshold:
                # Slower agent trailing = very bad
                speed_penalty = (1.0 - speed_ratio) * 0.5
                trailing_penalty += speed_penalty
                info["speed_penalty"] = speed_penalty

            total_penalty -= trailing_penalty

        # 3. Receding penalty: penalize moving away from target
        to_target = target_position - agent_position
        to_target_dist = np.linalg.norm(to_target)

        if to_target_dist > 1e-6:
            to_target_unit = to_target / to_target_dist
            approach_dot = np.dot(agent_dir, to_target_unit)

            if approach_dot < -0.3:  # Moving away
                receding_penalty = -approach_dot * 0.3  # Penalty for receding
                total_penalty -= receding_penalty
                info["receding_penalty"] = receding_penalty

        info["total_penalty"] = total_penalty
        return total_penalty, info


class InterceptGeometryBonus:
    """
    Rewards agents for achieving good intercept geometry.

    Optimal intercepts are perpendicular or head-on approaches,
    not tail-chase configurations.
    """

    def __init__(self, config: Optional[IntrinsicRewardConfig] = None):
        """Initialize intercept geometry bonus calculator."""
        self.config = config or IntrinsicRewardConfig()

    def compute(
        self,
        agent_position: np.ndarray,
        target_position: np.ndarray,
        target_velocity: np.ndarray,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute intercept geometry bonus.

        Args:
            agent_position: Agent position [x, y]
            target_position: Target position [x, y]
            target_velocity: Target velocity vector [vx, vy]

        Returns:
            bonus: Positive reward for good geometry
            info: Debug info dict
        """
        target_speed = np.linalg.norm(target_velocity)
        to_target = target_position - agent_position
        to_target_dist = np.linalg.norm(to_target)

        info = {
            "approach_angle": 0.0,
            "geometry_bonus": 0.0,
            "is_good_geometry": False,
        }

        if to_target_dist < 1e-6 or target_speed < 1e-6:
            return 0.0, info

        to_target_unit = to_target / to_target_dist
        target_dir = target_velocity / target_speed

        # Angle between approach direction and target's reverse direction
        # cos(angle) = dot(approach, -target_dir)
        approach_angle = np.arccos(np.clip(
            np.dot(to_target_unit, -target_dir), -1, 1
        ))
        info["approach_angle"] = approach_angle

        # Good geometry: > 45 degrees from pure tail-chase
        if approach_angle > self.config.good_approach_angle:
            bonus = self.config.geometry_bonus_scale * (approach_angle / np.pi)
            info["geometry_bonus"] = bonus
            info["is_good_geometry"] = True
            return bonus, info

        return 0.0, info


class NoveltySearch:
    """
    Novelty search via state embedding archive.

    Maintains archive of visited state embeddings and rewards
    states that are different from previously seen states.
    """

    def __init__(self, config: Optional[IntrinsicRewardConfig] = None):
        """Initialize novelty search."""
        self.config = config or IntrinsicRewardConfig()
        self.archive: deque = deque(maxlen=self.config.archive_max_size)
        self.embedding_dim = self.config.embedding_dim

    def _create_embedding(
        self,
        agent_position: np.ndarray,
        agent_velocity: np.ndarray,
        target_position: np.ndarray,
        relative_positions: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Create compact state embedding for novelty comparison.

        Args:
            agent_position: Agent position [x, y]
            agent_velocity: Agent velocity [vx, vy]
            target_position: Target position [x, y]
            relative_positions: Optional list of neighbor relative positions

        Returns:
            embedding: Compact state vector
        """
        # Core features
        rel_pos = target_position - agent_position
        distance = np.linalg.norm(rel_pos)
        angle = np.arctan2(rel_pos[1], rel_pos[0]) if distance > 0 else 0

        embedding = [
            agent_position[0] / 10000,  # Normalize to typical arena
            agent_position[1] / 10000,
            agent_velocity[0] / 500,  # Normalize to typical speeds
            agent_velocity[1] / 500,
            rel_pos[0] / 10000,
            rel_pos[1] / 10000,
            distance / 10000,
            np.sin(angle),
            np.cos(angle),
        ]

        # Add neighbor info if available
        if relative_positions:
            for i, rel_pos in enumerate(relative_positions[:3]):  # Up to 3 neighbors
                embedding.extend([
                    rel_pos[0] / 5000,
                    rel_pos[1] / 5000,
                ])

        # Pad or truncate to embedding_dim
        embedding = np.array(embedding[:self.embedding_dim])
        if len(embedding) < self.embedding_dim:
            embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))

        return embedding

    def compute_novelty(
        self,
        agent_position: np.ndarray,
        agent_velocity: np.ndarray,
        target_position: np.ndarray,
        relative_positions: Optional[List[np.ndarray]] = None,
        add_to_archive: bool = True,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute novelty bonus for current state.

        Args:
            agent_position: Agent position
            agent_velocity: Agent velocity
            target_position: Target position
            relative_positions: Optional neighbor relative positions
            add_to_archive: Whether to add this state to archive

        Returns:
            novelty_bonus: Reward inversely proportional to similarity to archive
            info: Debug info
        """
        embedding = self._create_embedding(
            agent_position, agent_velocity, target_position, relative_positions
        )

        info = {
            "archive_size": len(self.archive),
            "mean_distance": 0.0,
            "novelty_score": 0.0,
            "novelty_bonus": 0.0,
        }

        if len(self.archive) == 0:
            # First state is maximally novel
            if add_to_archive:
                self.archive.append(embedding)
            info["novelty_score"] = 1.0
            info["novelty_bonus"] = self.config.novelty_bonus_scale
            return self.config.novelty_bonus_scale, info

        # Compute distances to k-nearest neighbors
        archive_array = np.array(list(self.archive))
        distances = np.linalg.norm(archive_array - embedding, axis=1)

        k = min(self.config.novelty_k_neighbors, len(distances))
        k_nearest = np.partition(distances, k-1)[:k]
        mean_distance = np.mean(k_nearest)

        info["mean_distance"] = mean_distance

        # Novelty score: normalized mean distance
        # Higher distance = more novel
        novelty_score = np.tanh(mean_distance * 5)  # Saturate around 1
        info["novelty_score"] = novelty_score

        # Bonus proportional to novelty
        novelty_bonus = self.config.novelty_bonus_scale * novelty_score
        info["novelty_bonus"] = novelty_bonus

        # Add to archive
        if add_to_archive:
            self.archive.append(embedding)

        return novelty_bonus, info

    def reset(self):
        """Clear the novelty archive."""
        self.archive.clear()


class SwarmCoverageBonus:
    """
    Rewards swarm for achieving good spatial coverage.

    Encourages agents to spread out and cover the operational area.
    """

    def __init__(self, config: Optional[IntrinsicRewardConfig] = None):
        """Initialize swarm coverage bonus calculator."""
        self.config = config or IntrinsicRewardConfig()

    def compute(
        self,
        all_agent_positions: Dict[str, np.ndarray],
        target_position: np.ndarray,
        arena_size: float = 8000.0,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Compute coverage bonus for each agent.

        Args:
            all_agent_positions: Dict mapping agent_id to position
            target_position: Target position
            arena_size: Size of operational arena

        Returns:
            bonuses: Dict mapping agent_id to coverage bonus
            info: Global coverage info
        """
        positions = list(all_agent_positions.values())
        agent_ids = list(all_agent_positions.keys())

        if len(positions) < 2:
            return {aid: 0.0 for aid in agent_ids}, {"coverage_score": 0.0}

        positions_array = np.array(positions)

        # Compute pairwise distances
        dists = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dists.append(np.linalg.norm(positions_array[i] - positions_array[j]))

        mean_spacing = np.mean(dists) if dists else 0
        optimal_spacing = arena_size / np.sqrt(len(positions))  # Roughly uniform

        # Coverage score: how well agents are spread out
        coverage_ratio = mean_spacing / optimal_spacing if optimal_spacing > 0 else 0
        coverage_score = min(1.0, coverage_ratio)

        info = {
            "mean_spacing": mean_spacing,
            "optimal_spacing": optimal_spacing,
            "coverage_score": coverage_score,
            "num_agents": len(positions),
        }

        # Individual bonuses based on local density
        bonuses = {}
        for aid, pos in all_agent_positions.items():
            # Distance to nearest neighbor
            other_positions = [p for p in positions if not np.array_equal(p, pos)]
            if other_positions:
                min_dist = min(np.linalg.norm(pos - np.array(op)) for op in other_positions)
                # Bonus for good spacing (not too close, not too far)
                if min_dist > optimal_spacing * 0.3:
                    bonus = self.config.coverage_bonus_scale * coverage_score
                else:
                    bonus = 0.0  # Too close to another agent
            else:
                bonus = 0.0

            bonuses[aid] = bonus

        return bonuses, info


class IntrinsicRewardCalculator:
    """
    Combined intrinsic reward calculator.

    Aggregates all intrinsic reward components for use in training.
    """

    def __init__(self, config: Optional[IntrinsicRewardConfig] = None):
        """Initialize combined intrinsic reward calculator."""
        self.config = config or IntrinsicRewardConfig()

        self.velocity_penalty = VelocityMismatchPenalty(self.config)
        self.geometry_bonus = InterceptGeometryBonus(self.config)
        self.novelty_search = NoveltySearch(self.config)
        self.coverage_bonus = SwarmCoverageBonus(self.config)

        # Track detections for novelty
        self.detected_by: set = set()

    def compute_agent_reward(
        self,
        agent_id: str,
        agent_position: np.ndarray,
        agent_velocity: np.ndarray,
        target_position: np.ndarray,
        target_velocity: np.ndarray,
        detection_range: float,
        include_novelty: bool = True,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute total intrinsic reward for an agent.

        Args:
            agent_id: Agent identifier
            agent_position: Agent position
            agent_velocity: Agent velocity
            target_position: Target position
            target_velocity: Target velocity
            detection_range: Sensor range for detection bonus
            include_novelty: Whether to include novelty bonus

        Returns:
            total_reward: Combined intrinsic reward
            info: Component-wise breakdown
        """
        total = 0.0
        info = {}

        # 1. Anti-trailing penalty
        trailing_penalty, trailing_info = self.velocity_penalty.compute(
            agent_velocity, target_velocity, agent_position, target_position
        )
        total += trailing_penalty
        info["trailing"] = trailing_info

        # 2. Intercept geometry bonus
        geometry_bonus, geometry_info = self.geometry_bonus.compute(
            agent_position, target_position, target_velocity
        )
        total += geometry_bonus
        info["geometry"] = geometry_info

        # 3. Detection bonus
        distance_to_target = np.linalg.norm(target_position - agent_position)
        if distance_to_target <= detection_range:
            total += self.config.detection_bonus

            # First detection bonus
            if agent_id not in self.detected_by:
                total += self.config.new_detection_bonus
                self.detected_by.add(agent_id)
                info["new_detection"] = True

        info["distance_to_target"] = distance_to_target
        info["in_detection_range"] = distance_to_target <= detection_range

        # 4. Novelty bonus (optional, can be slow)
        if include_novelty:
            novelty_bonus, novelty_info = self.novelty_search.compute_novelty(
                agent_position, agent_velocity, target_position
            )
            total += novelty_bonus
            info["novelty"] = novelty_info

        info["total_intrinsic_reward"] = total
        return total, info

    def compute_swarm_rewards(
        self,
        agent_states: Dict[str, Dict],
        target_position: np.ndarray,
        target_velocity: np.ndarray,
        detection_range: float,
        arena_size: float = 8000.0,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Compute intrinsic rewards for entire swarm.

        Args:
            agent_states: Dict mapping agent_id to state dict with position/velocity
            target_position: Target position
            target_velocity: Target velocity
            detection_range: Sensor range
            arena_size: Arena size for coverage calculation

        Returns:
            rewards: Dict mapping agent_id to intrinsic reward
            info: Swarm-level info
        """
        rewards = {}
        agent_info = {}

        # Compute individual rewards
        for agent_id, state in agent_states.items():
            if not state.get("active", True):
                rewards[agent_id] = 0.0
                continue

            reward, info = self.compute_agent_reward(
                agent_id=agent_id,
                agent_position=state["position"],
                agent_velocity=state["velocity"],
                target_position=target_position,
                target_velocity=target_velocity,
                detection_range=detection_range,
                include_novelty=True,
            )
            rewards[agent_id] = reward
            agent_info[agent_id] = info

        # Compute coverage bonus
        positions = {
            aid: state["position"]
            for aid, state in agent_states.items()
            if state.get("active", True)
        }
        coverage_bonuses, coverage_info = self.coverage_bonus.compute(
            positions, target_position, arena_size
        )

        # Add coverage to individual rewards
        for agent_id, cov_bonus in coverage_bonuses.items():
            rewards[agent_id] = rewards.get(agent_id, 0.0) + cov_bonus

        swarm_info = {
            "agent_info": agent_info,
            "coverage": coverage_info,
            "total_intrinsic": sum(rewards.values()),
            "mean_intrinsic": np.mean(list(rewards.values())) if rewards else 0.0,
        }

        return rewards, swarm_info

    def reset(self):
        """Reset intrinsic reward state for new episode."""
        self.detected_by.clear()
        self.novelty_search.reset()
