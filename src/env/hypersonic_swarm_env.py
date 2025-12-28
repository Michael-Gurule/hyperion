"""
HYPERION Hypersonic Swarm Environment
PettingZoo Parallel API Implementation

This environment simulates a swarm of UAV agents coordinating to intercept
a hypersonic threat. Implements realistic physics, sensor models, and
multi-agent coordination challenges.
"""

import functools
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
from .visualization import SwarmVisualizer


class HypersonicSwarmEnv(ParallelEnv):
    """
    Multi-agent environment for hypersonic threat interception.

    Agents: UAV drones with limited fuel, sensors, and interceptor capability
    Target: Hypersonic vehicle traveling at Mach 5+ with predictable trajectory
    Goal: Coordinate to intercept target before it reaches protected zone

    State Space (per agent):
        - Own position (x, y) [normalized]
        - Own velocity (vx, vy) [normalized]
        - Fuel remaining [0-1]
        - Target position relative to agent (if detected)
        - Target velocity estimate (if detected)
        - Nearest neighbor positions (up to k neighbors)

    Action Space (per agent):
        - Thrust magnitude [0-1]
        - Heading change [-1, 1] (normalized angular acceleration)
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "hyperion_swarm_v0",
    }

    def __init__(
        self,
        num_agents: int = 5,
        max_steps: int = 500,
        arena_size: float = 10000.0,
        target_speed: float = 1700.0,
        agent_max_speed: float = 300.0,
        detection_range: float = 2000.0,
        intercept_range: float = 50.0,
        communication_range: float = 1500.0,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize hypersonic swarm environment.

        Args:
            num_agents: Number of UAV agents in swarm
            max_steps: Maximum timesteps before episode terminates
            arena_size: Size of operational area (square)
            target_speed: Hypersonic target velocity
            agent_max_speed: Maximum UAV velocity
            detection_range: Maximum sensor range for target detection
            intercept_range: Distance threshold for successful interception
            communication_range: Range for agent-to-agent communication
            render_mode: Visualization mode
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
        self.render_mode = render_mode

        # Time step (seconds)
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

        # Visualization
        self.visualizer = None
        if render_mode is not None:
            self.visualizer = SwarmVisualizer(
                arena_size=arena_size,
                detection_range=detection_range,
                communication_range=communication_range,
                intercept_range=intercept_range,
            )

        # Define observation and action spaces
        self._setup_spaces()

    def _setup_spaces(self):
        """Define observation and action spaces for agents."""

        max_neighbors = 3
        obs_dim = 5 + 5 + (max_neighbors * 4)

        self.observation_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
            for agent in self.possible_agents
        }

        self.action_spaces = {
            agent: spaces.Box(
                low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32
            )
            for agent in self.possible_agents
        }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """Return observation space for specified agent."""
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """Return action space for specified agent."""
        return self.action_spaces[agent]

    @property
    def num_agents(self):
        """Return number of agents."""
        return self._num_agents

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset environment to initial state.

        Returns:
            observations: Dict of observations for each agent
            infos: Dict of info dicts for each agent
        """
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self.current_step = 0

        # Initialize agent states
        self.agent_states = {}
        for i, agent in enumerate(self.agents):
            angle = 2 * np.pi * i / self._num_agents
            radius = self.arena_size * 0.3

            self.agent_states[agent] = {
                "position": np.array([radius * np.cos(angle), radius * np.sin(angle)]),
                "velocity": np.array([0.0, 0.0]),
                "heading": angle + np.pi,
                "fuel": self.initial_fuel,
                "active": True,
            }

        # Initialize hypersonic target
        entry_angle = np.random.uniform(0, 2 * np.pi)
        target_start = np.array(
            [
                self.arena_size * np.cos(entry_angle),
                self.arena_size * np.sin(entry_angle),
            ]
        )

        target_direction = -target_start / np.linalg.norm(target_start)
        noise = np.random.normal(0, 0.1, 2)
        target_direction = target_direction + noise
        target_direction = target_direction / np.linalg.norm(target_direction)

        self.target_state = {
            "position": target_start,
            "velocity": target_direction * self.target_speed,
            "active": True,
        }

        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        # Reset visualization
        if self.visualizer is not None:
            self.visualizer.reset_history()

        return observations, infos

    def step(self, actions: Dict[str, np.ndarray]):
        """
        Execute one timestep of environment dynamics.

        Args:
            actions: Dict mapping agent names to action arrays

        Returns:
            observations: Dict of observations for each agent
            rewards: Dict of rewards for each agent
            terminations: Dict of termination flags for each agent
            truncations: Dict of truncation flags for each agent
            infos: Dict of info dicts for each agent
        """
        self.current_step += 1

        # Update agent states based on actions
        for agent in self.agents:
            if not self.agent_states[agent]["active"]:
                continue

            action = actions.get(agent, np.array([0.0, 0.0]))
            self._update_agent(agent, action)

        # Update target state
        self._update_target()

        # Calculate observations and rewards
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        # Check interception
        intercepted = self._check_interception()
        target_escaped = self._check_target_escape()

        for agent in self.agents:
            observations[agent] = self._get_observation(agent)
            rewards[agent] = self._calculate_reward(agent, intercepted, target_escaped)
            terminations[agent] = (
                intercepted or target_escaped or not self.agent_states[agent]["active"]
            )
            truncations[agent] = self.current_step >= self.max_steps
            infos[agent] = {
                "intercepted": intercepted,
                "target_escaped": target_escaped,
                "fuel_remaining": self.agent_states[agent]["fuel"],
            }

        # Remove inactive agents
        if intercepted or target_escaped:
            self.agents = []

        # Record for visualization
        if self.visualizer is not None:
            self.visualizer.record_step(
                self.agent_states, self.target_state, intercepted
            )

        return observations, rewards, terminations, truncations, infos

    def _update_agent(self, agent: str, action: np.ndarray):
        """Update agent physics based on action."""
        state = self.agent_states[agent]

        thrust_magnitude = np.clip(action[0], 0.0, 1.0)
        heading_change = np.clip(action[1], -1.0, 1.0)

        # Update heading
        state["heading"] += heading_change * self.agent_max_turn_rate * self.dt
        state["heading"] = state["heading"] % (2 * np.pi)

        # Calculate acceleration
        thrust_direction = np.array(
            [np.cos(state["heading"]), np.sin(state["heading"])]
        )
        acceleration = thrust_direction * thrust_magnitude * self.agent_max_accel

        # Update velocity
        state["velocity"] += acceleration * self.dt

        # Enforce speed limit
        speed = np.linalg.norm(state["velocity"])
        if speed > self.agent_max_speed:
            state["velocity"] = state["velocity"] / speed * self.agent_max_speed

        # Update position
        state["position"] += state["velocity"] * self.dt

        # Update fuel
        state["fuel"] -= thrust_magnitude * self.fuel_consumption_rate * self.dt
        state["fuel"] = max(0.0, state["fuel"])

        # Deactivate if out of fuel
        if state["fuel"] <= 0.0:
            state["active"] = False

    def _update_target(self):
        """Update hypersonic target physics."""
        if not self.target_state["active"]:
            return

        # Simple ballistic trajectory
        self.target_state["position"] += self.target_state["velocity"] * self.dt

    def _get_observation(self, agent: str) -> np.ndarray:
        """
        Generate observation for specified agent.

        Observation includes:
        - Own state (position, velocity, fuel)
        - Target state (if detected)
        - Nearby agent states (if in communication range)
        """
        state = self.agent_states[agent]
        obs = []

        # Own state (normalized)
        obs.extend(state["position"] / self.arena_size)
        obs.extend(state["velocity"] / self.agent_max_speed)
        obs.append(state["fuel"])

        # Target detection
        target_pos = self.target_state["position"]
        distance_to_target = np.linalg.norm(state["position"] - target_pos)

        if distance_to_target <= self.detection_range:
            obs.append(1.0)
            obs.extend((target_pos - state["position"]) / self.detection_range)
            obs.extend(self.target_state["velocity"] / self.target_speed)
        else:
            obs.append(0.0)
            obs.extend([0.0, 0.0, 0.0, 0.0])

        # Neighboring agents
        neighbors = self._get_neighbors(agent, k=3)
        for neighbor in neighbors:
            neighbor_state = self.agent_states[neighbor]
            obs.extend(
                (neighbor_state["position"] - state["position"])
                / self.communication_range
            )
            obs.extend(neighbor_state["velocity"] / self.agent_max_speed)

        # Pad if fewer than 3 neighbors
        for _ in range(3 - len(neighbors)):
            obs.extend([0.0, 0.0, 0.0, 0.0])

        return np.array(obs, dtype=np.float32)

    def _get_neighbors(self, agent: str, k: int = 3) -> List[str]:
        """Get k nearest neighbors within communication range."""
        state = self.agent_states[agent]
        distances = []

        for other_agent in self.agents:
            if other_agent == agent or not self.agent_states[other_agent]["active"]:
                continue

            other_state = self.agent_states[other_agent]
            distance = np.linalg.norm(state["position"] - other_state["position"])

            if distance <= self.communication_range:
                distances.append((distance, other_agent))

        # Sort by distance and return k nearest
        distances.sort()
        return [agent for _, agent in distances[:k]]

    def _calculate_reward(self, agent: str, intercepted: bool, escaped: bool) -> float:
        """
        Calculate reward for agent.

        Reward components:
        - Large positive reward for successful interception
        - Large negative reward if target escapes
        - Small negative reward for distance to target
        - Small negative reward for fuel consumption
        - Small positive reward for formation maintenance
        """
        reward = 0.0
        state = self.agent_states[agent]

        if intercepted:
            reward += 100.0
        elif escaped:
            reward -= 100.0
        else:
            # Distance-based reward
            distance_to_target = np.linalg.norm(
                state["position"] - self.target_state["position"]
            )
            reward -= distance_to_target / self.arena_size * 0.1

            # Fuel efficiency
            reward -= (1.0 - state["fuel"]) * 0.01

            # Formation maintenance
            neighbors = self._get_neighbors(agent, k=3)
            if len(neighbors) > 0:
                avg_neighbor_distance = np.mean(
                    [
                        np.linalg.norm(
                            state["position"] - self.agent_states[n]["position"]
                        )
                        for n in neighbors
                    ]
                )
                if avg_neighbor_distance < self.communication_range * 0.5:
                    reward += 0.05

        return reward

    def _check_interception(self) -> bool:
        """Check if any agent successfully intercepted target."""
        if not self.target_state["active"]:
            return False

        target_pos = self.target_state["position"]

        for agent in self.agents:
            if not self.agent_states[agent]["active"]:
                continue

            distance = np.linalg.norm(self.agent_states[agent]["position"] - target_pos)

            if distance <= self.intercept_range:
                self.target_state["active"] = False
                return True

        return False

    def _check_target_escape(self) -> bool:
        """Check if target reached protected zone (origin)."""
        if not self.target_state["active"]:
            return False

        distance_to_origin = np.linalg.norm(self.target_state["position"])

        if distance_to_origin < 100.0:
            self.target_state["active"] = False
            return True

        return False

    def render(self):
        """Render environment."""
        if self.render_mode is None or self.visualizer is None:
            return

        fig = self.visualizer.render_frame(
            self.agent_states,
            self.target_state,
            self.current_step,
            show_detection=True,
            show_communication=True,
        )

        if self.render_mode == "human":
            plt.show()
        elif self.render_mode == "rgb_array":
            # Convert figure to RGB array
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return data

        plt.close(fig)

    def close(self):
        """Clean up environment resources."""
        pass


def env(**kwargs):
    """Factory function for wrapped environment."""
    env = HypersonicSwarmEnv(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env
