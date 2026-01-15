"""
Scaled Hypersonic Swarm Environment with enhanced features.

Supports 50-100+ agents with:
- Attention-based observation encoding for variable agent counts
- Adversarial target behavior with evasive maneuvers
- Coordination-aware reward shaping
- Communication failures and jamming simulation
- Projectile/missile system with PN guidance
"""

import functools
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from dataclasses import dataclass, field
from enum import Enum

from .projectile_system import ProjectileManager, ProjectileConfig, ProjectileHit


class TargetBehavior(Enum):
    """Target evasion behavior modes."""
    BALLISTIC = "ballistic"           # Straight line trajectory
    WEAVING = "weaving"               # Sinusoidal path
    RANDOM_JINK = "random_jink"       # Random direction changes
    PURSUIT_EVASION = "pursuit_evasion"  # Active evasion from pursuers
    TERRAIN_FOLLOWING = "terrain_following"  # Low altitude evasion


class AgentRole(Enum):
    """Agent roles for coordination."""
    SCOUT = 0      # Early detection, high speed
    TRACKER = 1    # Maintain target lock, moderate speed
    INTERCEPTOR = 2  # Intercept target, high maneuverability
    SUPPORT = 3    # Relay communications, backup


@dataclass
class AdversarialConfig:
    """Configuration for adversarial training."""
    enabled: bool = True
    evasive_maneuvers: bool = True
    evasion_probability: float = 0.3
    jink_frequency: float = 0.5  # Hz
    jink_magnitude: float = 500.0  # m/s^2
    jamming_enabled: bool = True
    jamming_probability: float = 0.1
    jamming_duration: float = 2.0  # seconds
    comm_failure_rate: float = 0.05


@dataclass
class RewardConfig:
    """Configuration for reward shaping."""
    # Terminal rewards
    intercept_reward: float = 100.0
    escape_penalty: float = -100.0

    # Distance rewards
    distance_scale: float = 0.1
    approach_bonus: float = 0.5  # Bonus for closing distance

    # Coordination rewards
    formation_bonus: float = 0.1
    role_bonus: float = 0.2
    coverage_bonus: float = 0.15

    # Efficiency penalties
    fuel_penalty: float = 0.01
    collision_penalty: float = -5.0

    # Curiosity/exploration
    detection_bonus: float = 0.3
    new_detection_bonus: float = 1.0

    # Projectile rewards
    projectile_hit_bonus: float = 50.0
    projectile_launch_cost: float = -0.5
    good_firing_angle_bonus: float = 2.0

    # Anti-trailing penalties
    trailing_penalty: float = 0.5
    intercept_geometry_bonus: float = 1.0


class ScaledHypersonicSwarmEnv(ParallelEnv):
    """
    Scaled multi-agent environment for hypersonic threat interception.
    Supports 50-100+ agents with efficient observation handling.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "hyperion_scaled_v1",
    }

    def __init__(
        self,
        num_agents: int = 50,
        max_steps: int = 500,
        arena_size: float = 8000.0,  # 8000x8000 arena per requirements
        target_speed: float = 1700.0,
        agent_max_speed: float = 300.0,
        detection_range: float = 3000.0,  # Extended for large swarms
        intercept_range: float = 100.0,  # Parameterized 50-200
        communication_range: float = 2000.0,  # Extended comm range
        max_observed_neighbors: int = 10,  # Attention over variable neighbors
        adversarial_config: Optional[AdversarialConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        projectile_config: Optional[ProjectileConfig] = None,
        render_mode: Optional[str] = None,
        # Projectile system toggle
        use_projectiles: bool = True,
        # Initial spread configuration
        initial_spread_min: float = 2000.0,
        initial_spread_max: float = 4000.0,
    ):
        """
        Initialize scaled swarm environment.

        Args:
            num_agents: Number of UAV agents (50-100+)
            max_steps: Maximum timesteps
            arena_size: Operational area size (default 8000x8000)
            target_speed: Hypersonic target velocity
            agent_max_speed: Maximum UAV velocity
            detection_range: Sensor range
            intercept_range: Interception threshold (50-200)
            communication_range: Agent communication range
            max_observed_neighbors: Max neighbors in observation
            adversarial_config: Adversarial training settings
            reward_config: Reward shaping configuration
            projectile_config: Projectile system configuration
            render_mode: Visualization mode
            use_projectiles: Enable projectile system
            initial_spread_min: Minimum initial drone spread radius
            initial_spread_max: Maximum initial drone spread radius
        """
        super().__init__()

        self._num_agents = num_agents
        self.max_steps = max_steps
        self.arena_size = arena_size
        self.target_speed = target_speed
        self.agent_max_speed = agent_max_speed
        self.detection_range = detection_range
        self.intercept_range = intercept_range
        self.communication_range = communication_range
        self.max_observed_neighbors = max_observed_neighbors
        self.render_mode = render_mode
        self.use_projectiles = use_projectiles
        self.initial_spread_min = initial_spread_min
        self.initial_spread_max = initial_spread_max

        # Configurations
        self.adversarial = adversarial_config or AdversarialConfig()
        self.rewards = reward_config or RewardConfig()

        # Projectile system
        if use_projectiles:
            self.projectile_config = projectile_config or ProjectileConfig()
            self.projectile_manager = ProjectileManager(self.projectile_config)
        else:
            self.projectile_config = None
            self.projectile_manager = None

        # Time step
        self.dt = 0.1

        # Agent identifiers
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents[:]

        # Physics constants
        self.agent_max_accel = 50.0
        self.agent_max_turn_rate = np.pi / 4
        self.initial_fuel = 1.0
        self.fuel_consumption_rate = 0.001

        # State tracking
        self.current_step = 0
        self.agent_states = {}
        self.target_state = None
        self.target_behavior = TargetBehavior.BALLISTIC

        # Jamming state
        self.jamming_active = False
        self.jamming_end_time = 0.0

        # Detection history for novelty bonus
        self.detected_by = set()

        # Projectile hit tracking for rewards
        self.projectile_hits_this_step = []
        self.agents_who_fired_this_step = set()

        # Define spaces
        self._setup_spaces()

    def _setup_spaces(self):
        """
        Define observation and action spaces.

        Observation structure:
        - Self state: 5 dims (pos_x, pos_y, vel_x, vel_y, fuel)
        - Target info: 6 dims (detected, rel_pos_x, rel_pos_y, vel_x, vel_y, confidence)
        - Neighbor embeddings: max_neighbors * 6 dims (rel_pos, vel, role, active)
        - Global context: 4 dims (num_active, mean_fuel, swarm_spread, time_remaining)
        - Projectile info: 4 dims (remaining_projectiles, cooldown, nearest_proj_x, nearest_proj_y)
        """
        self_dim = 5
        target_dim = 6
        neighbor_dim = 6
        global_dim = 4
        projectile_dim = 4 if self.use_projectiles else 0

        total_obs_dim = (
            self_dim +
            target_dim +
            self.max_observed_neighbors * neighbor_dim +
            global_dim +
            projectile_dim
        )

        self.observation_spaces = {
            agent: spaces.Box(
                low=-1.0, high=1.0, shape=(total_obs_dim,), dtype=np.float32
            )
            for agent in self.possible_agents
        }

        # Action space: [thrust, heading, fire] if projectiles enabled
        if self.use_projectiles:
            self.action_spaces = {
                agent: spaces.Box(
                    low=np.array([0.0, -1.0, 0.0]),
                    high=np.array([1.0, 1.0, 1.0]),
                    dtype=np.float32,
                )
                for agent in self.possible_agents
            }
        else:
            self.action_spaces = {
                agent: spaces.Box(
                    low=np.array([0.0, -1.0]),
                    high=np.array([1.0, 1.0]),
                    dtype=np.float32,
                )
                for agent in self.possible_agents
            }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    @property
    def num_agents(self):
        return self._num_agents

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
        """Reset environment."""
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self.current_step = 0
        self.detected_by = set()
        self.jamming_active = False
        self.projectile_hits_this_step = []
        self.agents_who_fired_this_step = set()

        # Reset projectile system
        if self.projectile_manager is not None:
            self.projectile_manager.reset()

        # Initialize agents in formation patterns
        self.agent_states = {}
        self._initialize_agents_formation()

        # Initialize target with random behavior
        self._initialize_target()

        # Select target behavior
        if self.adversarial.enabled and self.adversarial.evasive_maneuvers:
            behaviors = [
                TargetBehavior.BALLISTIC,
                TargetBehavior.WEAVING,
                TargetBehavior.RANDOM_JINK,
                TargetBehavior.PURSUIT_EVASION,
            ]
            self.target_behavior = np.random.choice(behaviors)
        else:
            self.target_behavior = TargetBehavior.BALLISTIC

        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: {"role": self._assign_role(agent)} for agent in self.agents}

        return observations, infos

    def _initialize_agents_formation(self):
        """Initialize agents in strategic formation with configurable spread."""
        num_agents = self._num_agents

        # Create multiple rings within the initial spread range
        num_rings = 3
        agents_per_ring = num_agents // num_rings

        # Use configurable spread range (2000-4000 default)
        spread_range = self.initial_spread_max - self.initial_spread_min
        ring_radii = [
            self.initial_spread_min + spread_range * 0.1,  # Inner ring (interceptors)
            self.initial_spread_min + spread_range * 0.5,  # Middle ring (trackers)
            self.initial_spread_min + spread_range * 0.9,  # Outer ring (scouts)
        ]

        agent_idx = 0
        for ring_idx, base_radius in enumerate(ring_radii):
            agents_in_ring = agents_per_ring if ring_idx < num_rings - 1 else num_agents - agent_idx

            for i in range(agents_in_ring):
                if agent_idx >= num_agents:
                    break

                angle = 2 * np.pi * i / agents_in_ring
                # Add small random perturbation
                angle += np.random.normal(0, 0.1)
                # Random radius within ring band
                radius = np.random.uniform(
                    base_radius * 0.9,
                    base_radius * 1.1
                )

                agent = self.possible_agents[agent_idx]
                self.agent_states[agent] = {
                    "position": np.array([
                        radius * np.cos(angle),
                        radius * np.sin(angle),
                    ]),
                    "velocity": np.array([0.0, 0.0]),
                    "heading": angle + np.pi,  # Face inward
                    "fuel": self.initial_fuel,
                    "active": True,
                    "role": AgentRole(ring_idx % 4),
                    "detection_count": 0,
                    "projectiles_launched": 0,  # Track for rewards
                }
                agent_idx += 1

    def _initialize_target(self):
        """Initialize hypersonic target."""
        # Entry from arena edge
        entry_angle = np.random.uniform(0, 2 * np.pi)
        target_start = np.array([
            self.arena_size * np.cos(entry_angle),
            self.arena_size * np.sin(entry_angle),
        ])

        # Direction toward center with noise
        target_direction = -target_start / np.linalg.norm(target_start)
        noise = np.random.normal(0, 0.1, 2)
        target_direction = target_direction + noise
        target_direction = target_direction / np.linalg.norm(target_direction)

        self.target_state = {
            "position": target_start,
            "velocity": target_direction * self.target_speed,
            "active": True,
            "jink_timer": 0.0,
            "current_jink": np.zeros(2),
        }

    def _assign_role(self, agent: str) -> AgentRole:
        """Get or assign role for agent."""
        if agent in self.agent_states:
            return self.agent_states[agent].get("role", AgentRole.INTERCEPTOR)
        return AgentRole.INTERCEPTOR

    def step(
        self,
        actions: Dict[str, np.ndarray],
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        """Execute environment step."""
        self.current_step += 1
        current_time = self.current_step * self.dt

        # Reset per-step tracking
        self.projectile_hits_this_step = []
        self.agents_who_fired_this_step = set()

        # Update jamming state
        self._update_jamming()

        # Update agents
        for agent in self.agents:
            if self.agent_states[agent]["active"]:
                default_action = np.array([0.0, 0.0, 0.0]) if self.use_projectiles else np.array([0.0, 0.0])
                action = actions.get(agent, default_action)
                self._update_agent(agent, action)

                # Handle projectile firing
                if self.use_projectiles and len(action) >= 3 and action[2] > 0.5:
                    self._try_fire_projectile(agent, current_time)

        # Update projectiles
        if self.projectile_manager is not None and self.target_state["active"]:
            hits = self.projectile_manager.update(
                dt=self.dt,
                current_time=current_time,
                target_position=self.target_state["position"],
                target_velocity=self.target_state["velocity"],
            )
            self.projectile_hits_this_step = hits

        # Update target with evasive behavior
        self._update_target()

        # Check collisions between agents
        self._check_collisions()

        # Calculate outcomes - projectile hits count as interception
        intercepted = self._check_interception()
        if not intercepted and self.projectile_hits_this_step:
            intercepted = True
            self.target_state["active"] = False
        target_escaped = self._check_target_escape()

        # Generate outputs
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        for agent in self.agents:
            observations[agent] = self._get_observation(agent)
            rewards[agent] = self._calculate_reward(agent, intercepted, target_escaped)
            terminations[agent] = (
                intercepted or
                target_escaped or
                not self.agent_states[agent]["active"]
            )
            truncations[agent] = self.current_step >= self.max_steps
            infos[agent] = {
                "intercepted": intercepted,
                "target_escaped": target_escaped,
                "fuel_remaining": self.agent_states[agent]["fuel"],
                "role": self.agent_states[agent]["role"],
                "jamming_active": self.jamming_active,
                "projectile_hits": len(self.projectile_hits_this_step),
                "fired_this_step": agent in self.agents_who_fired_this_step,
            }

        if intercepted or target_escaped:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _try_fire_projectile(self, agent: str, current_time: float):
        """Attempt to fire a projectile for an agent."""
        if self.projectile_manager is None:
            return

        state = self.agent_states[agent]
        if not state["active"]:
            return

        # Only fire if target is detected
        distance_to_target = np.linalg.norm(
            state["position"] - self.target_state["position"]
        )
        if distance_to_target > self.detection_range:
            return

        # Try to launch
        projectile = self.projectile_manager.launch_projectile(
            agent_id=agent,
            agent_position=state["position"],
            target_position=self.target_state["position"],
            target_velocity=self.target_state["velocity"],
            current_time=current_time,
        )

        if projectile is not None:
            self.agents_who_fired_this_step.add(agent)
            state["projectiles_launched"] = state.get("projectiles_launched", 0) + 1

    def _update_jamming(self):
        """Update jamming simulation."""
        if not self.adversarial.jamming_enabled:
            return

        current_time = self.current_step * self.dt

        if self.jamming_active:
            if current_time >= self.jamming_end_time:
                self.jamming_active = False
        else:
            if np.random.random() < self.adversarial.jamming_probability * self.dt:
                self.jamming_active = True
                self.jamming_end_time = current_time + self.adversarial.jamming_duration

    def _update_agent(self, agent: str, action: np.ndarray):
        """Update agent physics."""
        state = self.agent_states[agent]

        thrust_magnitude = np.clip(action[0], 0.0, 1.0)
        heading_change = np.clip(action[1], -1.0, 1.0)

        # Update heading
        state["heading"] += heading_change * self.agent_max_turn_rate * self.dt
        state["heading"] = state["heading"] % (2 * np.pi)

        # Acceleration
        thrust_direction = np.array([
            np.cos(state["heading"]),
            np.sin(state["heading"]),
        ])
        acceleration = thrust_direction * thrust_magnitude * self.agent_max_accel

        # Update velocity
        state["velocity"] += acceleration * self.dt

        # Enforce speed limit
        speed = np.linalg.norm(state["velocity"])
        if speed > self.agent_max_speed:
            state["velocity"] = state["velocity"] / speed * self.agent_max_speed

        # Update position
        state["position"] += state["velocity"] * self.dt

        # Fuel consumption
        state["fuel"] -= thrust_magnitude * self.fuel_consumption_rate * self.dt
        state["fuel"] = max(0.0, state["fuel"])

        if state["fuel"] <= 0.0:
            state["active"] = False

    def _update_target(self):
        """Update target with evasive behavior."""
        if not self.target_state["active"]:
            return

        target = self.target_state
        current_time = self.current_step * self.dt

        # Base velocity update
        base_velocity = target["velocity"].copy()

        # Apply evasive behavior
        if self.adversarial.enabled and self.adversarial.evasive_maneuvers:
            evasion = np.zeros(2)

            if self.target_behavior == TargetBehavior.WEAVING:
                # Sinusoidal weaving
                omega = self.adversarial.jink_frequency * 2 * np.pi
                perpendicular = np.array([-base_velocity[1], base_velocity[0]])
                perpendicular = perpendicular / (np.linalg.norm(perpendicular) + 1e-6)
                amplitude = self.adversarial.jink_magnitude * 0.1
                evasion = perpendicular * amplitude * np.sin(omega * current_time)

            elif self.target_behavior == TargetBehavior.RANDOM_JINK:
                # Random jinking
                target["jink_timer"] += self.dt
                if target["jink_timer"] >= 1.0 / self.adversarial.jink_frequency:
                    # New random jink direction
                    angle = np.random.uniform(0, 2 * np.pi)
                    target["current_jink"] = np.array([
                        np.cos(angle),
                        np.sin(angle),
                    ]) * self.adversarial.jink_magnitude * 0.1
                    target["jink_timer"] = 0.0
                evasion = target["current_jink"]

            elif self.target_behavior == TargetBehavior.PURSUIT_EVASION:
                # Active evasion from nearest pursuer
                nearest_dist = float("inf")
                nearest_pos = None

                for agent in self.agents:
                    if not self.agent_states[agent]["active"]:
                        continue
                    agent_pos = self.agent_states[agent]["position"]
                    dist = np.linalg.norm(agent_pos - target["position"])
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest_pos = agent_pos

                if nearest_pos is not None and nearest_dist < self.detection_range:
                    # Evade perpendicular to pursuer direction
                    to_pursuer = nearest_pos - target["position"]
                    perpendicular = np.array([-to_pursuer[1], to_pursuer[0]])
                    perpendicular = perpendicular / (np.linalg.norm(perpendicular) + 1e-6)
                    evasion_strength = min(1.0, self.detection_range / (nearest_dist + 1))
                    evasion = perpendicular * self.adversarial.jink_magnitude * evasion_strength * 0.1

            # Apply evasion with probability
            if np.random.random() < self.adversarial.evasion_probability:
                target["velocity"] = base_velocity + evasion * self.dt

        # Update position
        target["position"] += target["velocity"] * self.dt

        # Maintain speed
        speed = np.linalg.norm(target["velocity"])
        if abs(speed - self.target_speed) > 10.0:
            target["velocity"] = target["velocity"] / speed * self.target_speed

    def _check_collisions(self):
        """Check for agent-agent collisions."""
        collision_distance = 20.0  # meters

        agents_list = list(self.agents)
        for i, agent1 in enumerate(agents_list):
            if not self.agent_states[agent1]["active"]:
                continue

            pos1 = self.agent_states[agent1]["position"]

            for agent2 in agents_list[i + 1:]:
                if not self.agent_states[agent2]["active"]:
                    continue

                pos2 = self.agent_states[agent2]["position"]
                dist = np.linalg.norm(pos1 - pos2)

                if dist < collision_distance:
                    # Both agents damaged
                    self.agent_states[agent1]["fuel"] *= 0.5
                    self.agent_states[agent2]["fuel"] *= 0.5

    def _get_observation(self, agent: str) -> np.ndarray:
        """Generate observation with attention over neighbors."""
        state = self.agent_states[agent]
        obs = []

        # Self state (5 dims)
        obs.extend(state["position"] / self.arena_size)
        obs.extend(state["velocity"] / self.agent_max_speed)
        obs.append(state["fuel"])

        # Target detection (6 dims)
        target_pos = self.target_state["position"]
        distance_to_target = np.linalg.norm(state["position"] - target_pos)

        # Apply jamming effect
        effective_detection_range = self.detection_range
        if self.jamming_active:
            effective_detection_range *= 0.5  # Reduced range during jamming

        # Apply communication failure
        comm_working = np.random.random() > self.adversarial.comm_failure_rate

        if distance_to_target <= effective_detection_range:
            # Track detection
            self.detected_by.add(agent)
            state["detection_count"] += 1

            obs.append(1.0)  # Detected
            obs.extend((target_pos - state["position"]) / self.detection_range)
            obs.extend(self.target_state["velocity"] / self.target_speed)

            # Detection confidence (affected by distance and jamming)
            confidence = 1.0 - (distance_to_target / effective_detection_range)
            if self.jamming_active:
                confidence *= 0.7
            obs.append(confidence)
        else:
            obs.append(0.0)
            obs.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        # Neighbor observations (max_neighbors * 6 dims)
        neighbors = self._get_neighbors_with_attention(agent)
        for neighbor_data in neighbors:
            obs.extend(neighbor_data)

        # Pad if fewer neighbors
        neighbor_dim = 6
        for _ in range(self.max_observed_neighbors - len(neighbors)):
            obs.extend([0.0] * neighbor_dim)

        # Global context (4 dims)
        active_count = sum(
            1 for a in self.agents if self.agent_states[a]["active"]
        )
        mean_fuel = np.mean([
            self.agent_states[a]["fuel"]
            for a in self.agents if self.agent_states[a]["active"]
        ])

        # Swarm spread (variance of positions)
        positions = np.array([
            self.agent_states[a]["position"]
            for a in self.agents if self.agent_states[a]["active"]
        ])
        swarm_spread = np.var(positions) / (self.arena_size ** 2) if len(positions) > 1 else 0.0

        time_remaining = 1.0 - (self.current_step / self.max_steps)

        obs.append(active_count / self._num_agents)
        obs.append(mean_fuel)
        obs.append(min(1.0, swarm_spread))
        obs.append(time_remaining)

        # Projectile observations (4 dims) if enabled
        if self.use_projectiles and self.projectile_manager is not None:
            current_time = self.current_step * self.dt
            projectile_obs = self.projectile_manager.get_observation_for_agent(
                agent_id=agent,
                current_time=current_time,
                target_position=self.target_state["position"],
                normalize_distance=self.detection_range,
            )
            obs.extend(projectile_obs)

        return np.array(obs, dtype=np.float32)

    def _get_neighbors_with_attention(
        self,
        agent: str,
    ) -> List[List[float]]:
        """
        Get neighbors using attention-style prioritization.
        Prioritizes based on: distance, detection status, role diversity.
        """
        state = self.agent_states[agent]
        my_pos = state["position"]

        # Gather all neighbors in communication range
        candidates = []
        for other in self.agents:
            if other == agent or not self.agent_states[other]["active"]:
                continue

            other_state = self.agent_states[other]
            other_pos = other_state["position"]
            distance = np.linalg.norm(my_pos - other_pos)

            if distance <= self.communication_range:
                # Compute attention score
                distance_score = 1.0 - (distance / self.communication_range)

                # Bonus for agents with target detection
                detection_score = 0.5 if other_state["detection_count"] > 0 else 0.0

                # Role diversity bonus
                role_score = 0.3 if other_state["role"] != state["role"] else 0.0

                attention_score = distance_score + detection_score + role_score

                candidates.append({
                    "agent": other,
                    "distance": distance,
                    "score": attention_score,
                    "state": other_state,
                })

        # Sort by attention score and take top k
        candidates.sort(key=lambda x: x["score"], reverse=True)
        top_neighbors = candidates[:self.max_observed_neighbors]

        # Build neighbor features
        neighbor_features = []
        for candidate in top_neighbors:
            other_state = candidate["state"]

            # Relative position (2 dims)
            rel_pos = (other_state["position"] - my_pos) / self.communication_range

            # Relative velocity (2 dims)
            rel_vel = other_state["velocity"] / self.agent_max_speed

            # Role encoding (1 dim)
            role_encoding = other_state["role"].value / 3.0

            # Active status (1 dim)
            active = 1.0 if other_state["active"] else 0.0

            neighbor_features.append([
                rel_pos[0], rel_pos[1],
                rel_vel[0], rel_vel[1],
                role_encoding,
                active,
            ])

        return neighbor_features

    def _calculate_reward(
        self,
        agent: str,
        intercepted: bool,
        escaped: bool,
    ) -> float:
        """Calculate shaped reward for coordination with projectile and anti-trailing rewards."""
        reward = 0.0
        state = self.agent_states[agent]
        agent_vel = state["velocity"]
        agent_pos = state["position"]
        target_pos = self.target_state["position"]
        target_vel = self.target_state["velocity"]

        # Terminal rewards
        if intercepted:
            reward += self.rewards.intercept_reward
            # Bonus for agent who made the hit
            for hit in self.projectile_hits_this_step:
                if hit.owner_agent == agent:
                    reward += self.rewards.projectile_hit_bonus
            return reward
        elif escaped:
            reward += self.rewards.escape_penalty
            return reward

        # Distance-based reward
        distance_to_target = np.linalg.norm(agent_pos - target_pos)
        prev_distance = state.get("_prev_distance", distance_to_target)
        state["_prev_distance"] = distance_to_target

        # Reward for closing distance
        distance_delta = prev_distance - distance_to_target
        if distance_delta > 0:
            reward += self.rewards.approach_bonus * (distance_delta / 100.0)

        # Normalized distance penalty
        reward -= self.rewards.distance_scale * (distance_to_target / self.arena_size)

        # Detection bonus
        if distance_to_target <= self.detection_range:
            reward += self.rewards.detection_bonus

            # First detection bonus
            if agent not in self.detected_by:
                reward += self.rewards.new_detection_bonus
                self.detected_by.add(agent)

        # === ANTI-TRAILING PENALTIES ===
        agent_speed = np.linalg.norm(agent_vel)
        target_speed = np.linalg.norm(target_vel)

        if agent_speed > 1e-6 and target_speed > 1e-6:
            agent_dir = agent_vel / agent_speed
            target_dir = target_vel / target_speed

            # Penalize moving same direction as target (trailing)
            trailing_score = np.dot(agent_dir, target_dir)

            if trailing_score > 0.7:  # Moving same direction
                # Penalize more if slower than target
                speed_ratio = agent_speed / target_speed
                if speed_ratio < 0.8:
                    reward -= self.rewards.trailing_penalty * trailing_score

        # === INTERCEPT GEOMETRY BONUS ===
        # Reward positioning that creates good intercept angles
        to_target = target_pos - agent_pos
        to_target_dist = np.linalg.norm(to_target)

        if to_target_dist > 1e-6 and target_speed > 1e-6:
            to_target_unit = to_target / to_target_dist
            target_dir = target_vel / target_speed
            # Angle between approach direction and target's reverse direction
            approach_angle = np.arccos(np.clip(
                np.dot(to_target_unit, -target_dir), -1, 1
            ))
            # Optimal intercept is perpendicular or head-on (>45 degrees from behind)
            if approach_angle > np.pi / 4:  # > 45 degrees from tail-chase
                reward += self.rewards.intercept_geometry_bonus * (approach_angle / np.pi)

        # === PROJECTILE REWARDS ===
        if self.use_projectiles:
            # Penalty for firing (prevents spam)
            if agent in self.agents_who_fired_this_step:
                reward += self.rewards.projectile_launch_cost

            # Bonus for good firing angle (perpendicular or head-on approach)
            if agent in self.agents_who_fired_this_step and to_target_dist > 1e-6:
                if target_speed > 1e-6:
                    target_dir_fire = target_vel / target_speed
                    firing_angle = np.arccos(np.clip(
                        np.dot(to_target / to_target_dist, -target_dir_fire), -1, 1
                    ))
                    # Better angle = more bonus (head-on is best)
                    if firing_angle > np.pi / 3:  # > 60 degrees from behind
                        reward += self.rewards.good_firing_angle_bonus * (firing_angle / np.pi)

        # Coordination rewards
        neighbors = self._get_neighbors_with_attention(agent)

        if len(neighbors) > 0:
            # Formation maintenance
            neighbor_distances = [
                np.linalg.norm(
                    state["position"] - self.agent_states[other]["position"]
                )
                for other in self.agents
                if other != agent and self.agent_states[other]["active"]
                and np.linalg.norm(state["position"] - self.agent_states[other]["position"]) < self.communication_range
            ][:5]  # Top 5 closest

            if neighbor_distances:
                avg_neighbor_distance = np.mean(neighbor_distances)
                optimal_spacing = self.communication_range * 0.3
                spacing_error = abs(avg_neighbor_distance - optimal_spacing) / optimal_spacing

                if spacing_error < 0.5:
                    reward += self.rewards.formation_bonus * (1.0 - spacing_error)

            # Role-based coordination
            my_role = state["role"]

            if my_role == AgentRole.SCOUT:
                # Scouts rewarded for being far and detecting
                if distance_to_target > self.detection_range * 0.8:
                    reward += self.rewards.role_bonus * 0.5
                if state["detection_count"] > 0:
                    reward += self.rewards.role_bonus

            elif my_role == AgentRole.INTERCEPTOR:
                # Interceptors rewarded for being close to target
                if distance_to_target < self.detection_range * 0.5:
                    reward += self.rewards.role_bonus

            elif my_role == AgentRole.TRACKER:
                # Trackers rewarded for maintaining medium distance
                if self.detection_range * 0.3 < distance_to_target < self.detection_range * 0.7:
                    reward += self.rewards.role_bonus

        # Coverage bonus (spread out to cover more area)
        positions = np.array([
            self.agent_states[a]["position"]
            for a in self.agents if self.agent_states[a]["active"]
        ])
        if len(positions) > 5:
            # Compute convex hull area approximation
            spread = np.std(positions, axis=0).mean()
            normalized_spread = spread / (self.arena_size * 0.3)
            if normalized_spread > 0.5:
                reward += self.rewards.coverage_bonus * min(1.0, normalized_spread)

        # Efficiency penalties
        reward -= self.rewards.fuel_penalty * (1.0 - state["fuel"])

        return reward

    def _check_interception(self) -> bool:
        """Check for successful interception."""
        if not self.target_state["active"]:
            return False

        target_pos = self.target_state["position"]

        for agent in self.agents:
            if not self.agent_states[agent]["active"]:
                continue

            distance = np.linalg.norm(
                self.agent_states[agent]["position"] - target_pos
            )

            if distance <= self.intercept_range:
                self.target_state["active"] = False
                return True

        return False

    def _check_target_escape(self) -> bool:
        """Check if target escaped to protected zone."""
        if not self.target_state["active"]:
            return False

        distance_to_origin = np.linalg.norm(self.target_state["position"])

        if distance_to_origin < 100.0:
            self.target_state["active"] = False
            return True

        return False

    def render(self):
        """Render environment (placeholder)."""
        pass

    def close(self):
        """Clean up resources."""
        pass


def create_scaled_env(
    num_agents: int = 50,
    adversarial: bool = True,
    **kwargs,
) -> ScaledHypersonicSwarmEnv:
    """
    Factory function for scaled environment.

    Args:
        num_agents: Number of agents (50-100+)
        adversarial: Enable adversarial training
        **kwargs: Additional environment parameters

    Returns:
        Configured environment
    """
    adversarial_config = AdversarialConfig(enabled=adversarial)
    reward_config = RewardConfig()

    env = ScaledHypersonicSwarmEnv(
        num_agents=num_agents,
        adversarial_config=adversarial_config,
        reward_config=reward_config,
        **kwargs,
    )

    return env
