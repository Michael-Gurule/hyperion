"""
RLlib-compatible wrapper for HYPERION environment.
Converts PettingZoo ParallelEnv to RLlib's MultiAgentEnv format.
"""

from typing import Dict, Tuple, Optional, Any
import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from .hypersonic_swarm_env import HypersonicSwarmEnv


class RLlibHyperionEnv(MultiAgentEnv):
    """
    Wrapper to make HypersonicSwarmEnv compatible with RLlib's MultiAgentEnv API.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize wrapped environment.

        Args:
            config: Environment configuration dictionary
        """
        super().__init__()

        # Use config if provided, otherwise use defaults
        if config is None:
            config = {}

        # Create base environment
        self.env = HypersonicSwarmEnv(**config)

        # Set required attributes
        self._agent_ids = set(self.env.possible_agents)
        self._spaces_in_preferred_format = True

        # Store observation and action spaces as properties (not methods)
        self._observation_space = {
            agent: self.env.observation_space(agent)
            for agent in self.env.possible_agents
        }
        self._action_space = {
            agent: self.env.action_space(agent) for agent in self.env.possible_agents
        }

    @property
    def observation_space(self) -> Dict[str, spaces.Space]:
        """Return observation space dictionary."""
        return self._observation_space

    @property
    def action_space(self) -> Dict[str, spaces.Space]:
        """Return action space dictionary."""
        return self._action_space

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        """
        Reset the environment.

        Returns:
            observations: Dict of observations for each agent
            infos: Dict of info dicts for each agent
        """
        obs, infos = self.env.reset(seed=seed, options=options)
        return obs, infos

    def step(self, action_dict: Dict) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Step the environment.

        Args:
            action_dict: Dictionary mapping agent IDs to actions

        Returns:
            observations: Dict of observations
            rewards: Dict of rewards
            terminateds: Dict of termination flags
            truncateds: Dict of truncation flags
            infos: Dict of info dicts
        """
        obs, rewards, terminateds, truncateds, infos = self.env.step(action_dict)

        # RLlib expects "__all__" key for global termination/truncation
        terminateds["__all__"] = all(terminateds.values()) or len(self.env.agents) == 0
        truncateds["__all__"] = all(truncateds.values()) or len(self.env.agents) == 0

        return obs, rewards, terminateds, truncateds, infos

    def get_agent_ids(self) -> set:
        """Return set of agent IDs."""
        return self._agent_ids

    def render(self):
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Close the environment."""
        self.env.close()
