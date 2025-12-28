"""
Multi-Agent PPO (MAPPO) implementation with centralized critic.

MAPPO uses a centralized value function during training while maintaining
decentralized execution, improving credit assignment for multi-agent coordination.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class MAPPOConfig:
    """Configuration for MAPPO algorithm."""

    # Learning rates
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3

    # PPO parameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    clip_value: bool = True
    value_clip_param: float = 10.0

    # Entropy
    entropy_coeff: float = 0.01
    entropy_decay: float = 0.9999
    min_entropy_coeff: float = 0.001

    # Training
    num_epochs: int = 10
    num_minibatches: int = 4
    max_grad_norm: float = 0.5

    # Network
    hidden_dim: int = 256
    num_layers: int = 2
    use_gnn: bool = True
    gnn_layers: int = 3

    # Centralized critic
    use_centralized_critic: bool = True
    share_actor: bool = True

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ActorNetwork(nn.Module):
    """
    Actor network for decentralized policy.
    Each agent uses its local observation to select actions.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        """
        Initialize actor network.

        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Build network
        layers = [nn.Linear(obs_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ])

        self.backbone = nn.Sequential(*layers)

        # Policy head (Gaussian)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to get action distribution parameters.

        Args:
            obs: Agent observation [batch, obs_dim]

        Returns:
            mean: Action mean [batch, action_dim]
            std: Action std [action_dim]
        """
        features = self.backbone(obs)
        mean = self.mean_head(features)
        std = torch.exp(torch.clamp(self.log_std, -5, 2))
        return mean, std

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            obs: Observation
            deterministic: Use mean action

        Returns:
            action: Sampled action
            log_prob: Log probability
        """
        mean, std = self.forward(obs)

        if deterministic:
            action = mean
            log_prob = torch.zeros(obs.size(0), device=obs.device)
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

        # Apply action bounds
        action = torch.tanh(action)
        action = self._transform_action(action)

        return action, log_prob

    def evaluate(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy for given actions.

        Args:
            obs: Observations
            action: Actions taken

        Returns:
            log_prob: Log probabilities
            entropy: Policy entropy
        """
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)

        # Inverse transform action for log_prob
        action_transformed = self._inverse_transform_action(action)

        log_prob = dist.log_prob(action_transformed).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy

    def _transform_action(self, action: torch.Tensor) -> torch.Tensor:
        """Transform action from tanh output to valid range."""
        transformed = action.clone()
        transformed[:, 0] = (action[:, 0] + 1) / 2  # Thrust: [-1,1] -> [0,1]
        return transformed

    def _inverse_transform_action(self, action: torch.Tensor) -> torch.Tensor:
        """Inverse transform for log_prob computation."""
        transformed = action.clone()
        transformed[:, 0] = action[:, 0] * 2 - 1  # Thrust: [0,1] -> [-1,1]
        return transformed


class CentralizedCritic(nn.Module):
    """
    Centralized critic that observes all agents' states.
    Uses attention to aggregate global information efficiently.
    """

    def __init__(
        self,
        obs_dim: int,
        max_agents: int = 100,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
    ):
        """
        Initialize centralized critic.

        Args:
            obs_dim: Per-agent observation dimension
            max_agents: Maximum number of agents
            hidden_dim: Hidden layer dimension
            num_layers: Number of processing layers
            num_heads: Attention heads
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        # Per-agent encoder
        self.agent_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Self-attention for global context
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        all_obs: torch.Tensor,
        agent_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute centralized value for each agent.

        Args:
            all_obs: All agent observations [batch, num_agents, obs_dim]
            agent_mask: Mask for valid agents [batch, num_agents]

        Returns:
            values: Value estimate per agent [batch, num_agents]
        """
        batch_size, num_agents, _ = all_obs.shape

        # Encode each agent
        agent_features = self.agent_encoder(all_obs)  # [B, N, hidden]

        # Apply attention layers
        features = agent_features
        for attn, norm in zip(self.attention_layers, self.layer_norms):
            # Self-attention with residual
            attn_out, _ = attn(features, features, features, key_padding_mask=agent_mask)
            features = norm(features + attn_out)

        # Per-agent value
        values = self.value_head(features).squeeze(-1)  # [B, N]

        return values


class GlobalCritic(nn.Module):
    """
    Alternative: Global critic using mean aggregation.
    More efficient for very large swarms (100+ agents).
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        """
        Initialize global critic.

        Args:
            obs_dim: Per-agent observation dimension
            hidden_dim: Hidden dimension
            num_layers: Number of layers
        """
        super().__init__()

        # Agent encoder
        self.agent_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Global aggregation layers
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
            ])
        self.global_layers = nn.Sequential(*layers)

        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        agent_obs: torch.Tensor,
        all_obs: torch.Tensor,
        agent_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute value for specific agent with global context.

        Args:
            agent_obs: This agent's observation [batch, obs_dim]
            all_obs: All agent observations [batch, num_agents, obs_dim]
            agent_mask: Mask for valid agents [batch, num_agents]

        Returns:
            value: Value estimate [batch]
        """
        # Encode this agent
        agent_feat = self.agent_encoder(agent_obs)  # [B, hidden]

        # Encode all agents and aggregate
        all_feat = self.agent_encoder(all_obs)  # [B, N, hidden]

        if agent_mask is not None:
            # Masked mean pooling
            mask = agent_mask.unsqueeze(-1).float()
            global_feat = (all_feat * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        else:
            global_feat = all_feat.mean(dim=1)  # [B, hidden]

        # Combine local and global
        combined = torch.cat([agent_feat, global_feat], dim=-1)
        features = self.global_layers(combined)

        value = self.value_head(features).squeeze(-1)

        return value


@dataclass
class RolloutBuffer:
    """Storage for rollout data."""

    observations: List[Dict[str, np.ndarray]] = field(default_factory=list)
    actions: List[Dict[str, np.ndarray]] = field(default_factory=list)
    rewards: List[Dict[str, float]] = field(default_factory=list)
    dones: List[Dict[str, bool]] = field(default_factory=list)
    log_probs: List[Dict[str, float]] = field(default_factory=list)
    values: List[Dict[str, float]] = field(default_factory=list)

    def clear(self):
        """Clear buffer."""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def __len__(self) -> int:
        return len(self.observations)


class MAPPO:
    """
    Multi-Agent PPO with centralized training and decentralized execution.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_agents: int,
        config: Optional[MAPPOConfig] = None,
    ):
        """
        Initialize MAPPO.

        Args:
            obs_dim: Per-agent observation dimension
            action_dim: Action dimension
            num_agents: Number of agents
            config: MAPPO configuration
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.config = config or MAPPOConfig()

        self.device = torch.device(self.config.device)

        # Create actor (shared or independent)
        if self.config.share_actor:
            self.actor = ActorNetwork(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.num_layers,
            ).to(self.device)
            self.actors = {"shared": self.actor}
        else:
            self.actors = {}
            for i in range(num_agents):
                self.actors[f"agent_{i}"] = ActorNetwork(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    hidden_dim=self.config.hidden_dim,
                    num_layers=self.config.num_layers,
                ).to(self.device)

        # Create centralized critic
        if self.config.use_centralized_critic:
            self.critic = CentralizedCritic(
                obs_dim=obs_dim,
                max_agents=num_agents,
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.num_layers,
            ).to(self.device)
        else:
            self.critic = GlobalCritic(
                obs_dim=obs_dim,
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.num_layers,
            ).to(self.device)

        # Optimizers
        actor_params = []
        for actor in self.actors.values():
            actor_params.extend(actor.parameters())

        self.actor_optimizer = torch.optim.Adam(
            actor_params, lr=self.config.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.critic_lr
        )

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # Training stats
        self.entropy_coeff = self.config.entropy_coeff
        self.training_step = 0

    def get_actor(self, agent_id: str) -> ActorNetwork:
        """Get actor network for agent."""
        if self.config.share_actor:
            return self.actors["shared"]
        return self.actors.get(agent_id, self.actors["shared"])

    def select_actions(
        self,
        observations: Dict[str, np.ndarray],
        deterministic: bool = False,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, float]]:
        """
        Select actions for all agents.

        Args:
            observations: Dict of agent observations
            deterministic: Use mean actions

        Returns:
            actions: Dict of actions
            log_probs: Dict of log probabilities
            values: Dict of value estimates
        """
        actions = {}
        log_probs = {}
        values = {}

        # Convert observations to tensors
        agent_ids = sorted(observations.keys())
        obs_list = [observations[aid] for aid in agent_ids]
        all_obs = torch.FloatTensor(np.stack(obs_list)).to(self.device)

        # Get values from centralized critic
        if self.config.use_centralized_critic:
            all_obs_batch = all_obs.unsqueeze(0)  # [1, N, obs_dim]
            all_values = self.critic(all_obs_batch).squeeze(0)  # [N]
        else:
            all_values = None

        # Get actions from actors
        for i, agent_id in enumerate(agent_ids):
            obs = torch.FloatTensor(observations[agent_id]).unsqueeze(0).to(self.device)
            actor = self.get_actor(agent_id)

            with torch.no_grad():
                action, log_prob = actor.get_action(obs, deterministic)

            actions[agent_id] = action.squeeze(0).cpu().numpy()
            log_probs[agent_id] = log_prob.item()

            if all_values is not None:
                values[agent_id] = all_values[i].item()
            else:
                # Use global critic
                value = self.critic(obs.squeeze(0), all_obs.unsqueeze(0))
                values[agent_id] = value.item()

        return actions, log_probs, values

    def store_transition(
        self,
        observations: Dict[str, np.ndarray],
        actions: Dict[str, np.ndarray],
        rewards: Dict[str, float],
        dones: Dict[str, bool],
        log_probs: Dict[str, float],
        values: Dict[str, float],
    ):
        """Store transition in buffer."""
        self.buffer.observations.append(observations)
        self.buffer.actions.append(actions)
        self.buffer.rewards.append(rewards)
        self.buffer.dones.append(dones)
        self.buffer.log_probs.append(log_probs)
        self.buffer.values.append(values)

    def compute_returns_and_advantages(
        self,
        last_values: Dict[str, float],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Compute returns and GAE advantages.

        Args:
            last_values: Value estimates for last state

        Returns:
            returns: Return for each transition
            advantages: GAE advantages
        """
        agent_ids = list(self.buffer.observations[0].keys())
        num_steps = len(self.buffer)

        returns = {aid: np.zeros(num_steps) for aid in agent_ids}
        advantages = {aid: np.zeros(num_steps) for aid in agent_ids}

        for agent_id in agent_ids:
            last_gae = 0.0
            last_value = last_values[agent_id]

            for t in reversed(range(num_steps)):
                reward = self.buffer.rewards[t][agent_id]
                done = self.buffer.dones[t][agent_id]
                value = self.buffer.values[t][agent_id]

                if done:
                    next_value = 0.0
                    last_gae = 0.0
                elif t == num_steps - 1:
                    next_value = last_value
                else:
                    next_value = self.buffer.values[t + 1][agent_id]

                delta = reward + self.config.gamma * next_value - value
                last_gae = delta + self.config.gamma * self.config.gae_lambda * last_gae

                advantages[agent_id][t] = last_gae
                returns[agent_id][t] = last_gae + value

        return returns, advantages

    def update(
        self,
        last_values: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Perform PPO update.

        Args:
            last_values: Value estimates for terminal state

        Returns:
            Training statistics
        """
        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages(last_values)

        # Flatten data for all agents
        all_obs = []
        all_actions = []
        all_old_log_probs = []
        all_returns = []
        all_advantages = []
        all_old_values = []
        all_agent_ids = []

        agent_ids = list(self.buffer.observations[0].keys())
        num_steps = len(self.buffer)

        for t in range(num_steps):
            for agent_id in agent_ids:
                all_obs.append(self.buffer.observations[t][agent_id])
                all_actions.append(self.buffer.actions[t][agent_id])
                all_old_log_probs.append(self.buffer.log_probs[t][agent_id])
                all_returns.append(returns[agent_id][t])
                all_advantages.append(advantages[agent_id][t])
                all_old_values.append(self.buffer.values[t][agent_id])
                all_agent_ids.append(agent_id)

        # Convert to tensors
        obs_tensor = torch.FloatTensor(np.array(all_obs)).to(self.device)
        actions_tensor = torch.FloatTensor(np.array(all_actions)).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(all_old_log_probs).to(self.device)
        returns_tensor = torch.FloatTensor(all_returns).to(self.device)
        advantages_tensor = torch.FloatTensor(all_advantages).to(self.device)
        old_values_tensor = torch.FloatTensor(all_old_values).to(self.device)

        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
            advantages_tensor.std() + 1e-8
        )

        # Training loop
        total_samples = len(all_obs)
        batch_size = total_samples // self.config.num_minibatches

        stats = defaultdict(list)

        for epoch in range(self.config.num_epochs):
            # Shuffle indices
            indices = np.random.permutation(total_samples)

            for start in range(0, total_samples, batch_size):
                end = min(start + batch_size, total_samples)
                batch_indices = indices[start:end]

                # Get batch data
                batch_obs = obs_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_old_values = old_values_tensor[batch_indices]

                # Evaluate current policy
                actor = self.get_actor(agent_ids[0])  # Shared actor
                new_log_probs, entropy = actor.evaluate(batch_obs, batch_actions)

                # Policy loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1 - self.config.clip_param,
                    1 + self.config.clip_param,
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Total actor loss
                actor_loss = policy_loss + self.entropy_coeff * entropy_loss

                # Value loss
                # For centralized critic, we need to gather global observations
                # Simplified: use batch observations directly
                new_values = self._get_batch_values(batch_obs, batch_indices, agent_ids)

                if self.config.clip_value:
                    value_clipped = batch_old_values + torch.clamp(
                        new_values - batch_old_values,
                        -self.config.value_clip_param,
                        self.config.value_clip_param,
                    )
                    value_loss1 = (new_values - batch_returns) ** 2
                    value_loss2 = (value_clipped - batch_returns) ** 2
                    value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                else:
                    value_loss = 0.5 * ((new_values - batch_returns) ** 2).mean()

                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actors.values())[0].parameters(),
                    self.config.max_grad_norm,
                )
                self.actor_optimizer.step()

                # Update critic
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(),
                    self.config.max_grad_norm,
                )
                self.critic_optimizer.step()

                # Log stats
                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(-entropy_loss.item())
                stats["approx_kl"].append(
                    (batch_old_log_probs - new_log_probs).mean().item()
                )
                stats["clip_fraction"].append(
                    ((ratio - 1).abs() > self.config.clip_param).float().mean().item()
                )

        # Decay entropy coefficient
        self.entropy_coeff = max(
            self.config.min_entropy_coeff,
            self.entropy_coeff * self.config.entropy_decay,
        )

        # Clear buffer
        self.buffer.clear()
        self.training_step += 1

        # Average stats
        return {k: np.mean(v) for k, v in stats.items()}

    def _get_batch_values(
        self,
        batch_obs: torch.Tensor,
        batch_indices: np.ndarray,
        agent_ids: List[str],
    ) -> torch.Tensor:
        """
        Get value estimates for batch.
        Handles centralized critic with global observations.
        """
        if self.config.use_centralized_critic:
            # Need to reconstruct global observations for each sample
            # Simplified: treat batch as single environment
            batch_size = batch_obs.size(0)
            num_agents = len(agent_ids)

            # Pad to create [batch, num_agents, obs_dim]
            # This is approximate - full implementation would track per-timestep
            all_obs = batch_obs.unsqueeze(1).expand(-1, num_agents, -1)
            values = self.critic(all_obs)[:, 0]  # Take first agent's value
            return values
        else:
            # Global critic with mean aggregation
            global_context = batch_obs.unsqueeze(1).expand(-1, 1, -1)
            values = self.critic(batch_obs, global_context)
            return values

    def save(self, path: str):
        """Save model checkpoints."""
        checkpoint = {
            "actors": {k: v.state_dict() for k, v in self.actors.items()},
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "entropy_coeff": self.entropy_coeff,
            "training_step": self.training_step,
            "config": self.config,
        }
        torch.save(checkpoint, path)

    def load(self, path: str):
        """Load model checkpoints."""
        checkpoint = torch.load(path, map_location=self.device)

        for k, v in checkpoint["actors"].items():
            if k in self.actors:
                self.actors[k].load_state_dict(v)

        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.entropy_coeff = checkpoint["entropy_coeff"]
        self.training_step = checkpoint["training_step"]


class MAPPOTrainer:
    """
    High-level trainer for MAPPO with curriculum learning support.
    """

    def __init__(
        self,
        env,
        config: Optional[MAPPOConfig] = None,
    ):
        """
        Initialize MAPPO trainer.

        Args:
            env: PettingZoo parallel environment
            config: MAPPO configuration
        """
        self.env = env
        self.config = config or MAPPOConfig()

        # Get environment info
        sample_agent = env.possible_agents[0]
        obs_dim = env.observation_space(sample_agent).shape[0]
        action_dim = env.action_space(sample_agent).shape[0]
        num_agents = len(env.possible_agents)

        # Create MAPPO
        self.mappo = MAPPO(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_agents=num_agents,
            config=config,
        )

        # Training stats
        self.episode_rewards = []
        self.episode_lengths = []

    def train(
        self,
        total_timesteps: int,
        rollout_length: int = 2048,
        log_interval: int = 1,
        eval_interval: int = 10,
        checkpoint_interval: int = 100,
        checkpoint_path: str = "./checkpoints",
    ) -> Dict[str, List[float]]:
        """
        Train MAPPO agent.

        Args:
            total_timesteps: Total training timesteps
            rollout_length: Steps per rollout
            log_interval: Episodes between logging
            eval_interval: Episodes between evaluation
            checkpoint_interval: Episodes between checkpoints
            checkpoint_path: Path for checkpoints

        Returns:
            Training history
        """
        import os
        os.makedirs(checkpoint_path, exist_ok=True)

        timesteps = 0
        episodes = 0
        history = defaultdict(list)

        while timesteps < total_timesteps:
            # Collect rollout
            observations, _ = self.env.reset()
            episode_reward = {aid: 0.0 for aid in observations.keys()}
            episode_length = 0

            for _ in range(rollout_length):
                # Select actions
                actions, log_probs, values = self.mappo.select_actions(observations)

                # Step environment
                next_obs, rewards, terminations, truncations, infos = self.env.step(actions)

                # Check if done
                dones = {
                    aid: terminations.get(aid, False) or truncations.get(aid, False)
                    for aid in observations.keys()
                }

                # Store transition
                self.mappo.store_transition(
                    observations, actions, rewards, dones, log_probs, values
                )

                # Update stats
                for aid in observations.keys():
                    episode_reward[aid] += rewards.get(aid, 0.0)
                episode_length += 1
                timesteps += len(observations)

                # Check termination
                if all(dones.values()) or len(next_obs) == 0:
                    # Episode ended
                    episodes += 1
                    self.episode_rewards.append(np.mean(list(episode_reward.values())))
                    self.episode_lengths.append(episode_length)

                    # Reset
                    observations, _ = self.env.reset()
                    episode_reward = {aid: 0.0 for aid in observations.keys()}
                    episode_length = 0

                    # Log
                    if episodes % log_interval == 0:
                        mean_reward = np.mean(self.episode_rewards[-log_interval:])
                        mean_length = np.mean(self.episode_lengths[-log_interval:])
                        print(
                            f"Episode {episodes} | "
                            f"Timesteps {timesteps} | "
                            f"Mean Reward: {mean_reward:.2f} | "
                            f"Mean Length: {mean_length:.1f}"
                        )
                        history["episode"].append(episodes)
                        history["mean_reward"].append(mean_reward)
                        history["mean_length"].append(mean_length)
                else:
                    observations = next_obs

            # Get terminal values
            _, _, last_values = self.mappo.select_actions(observations)

            # Update policy
            update_stats = self.mappo.update(last_values)

            # Log update stats
            history["policy_loss"].append(update_stats["policy_loss"])
            history["value_loss"].append(update_stats["value_loss"])
            history["entropy"].append(update_stats["entropy"])

            # Checkpoint
            if episodes % checkpoint_interval == 0:
                self.mappo.save(
                    os.path.join(checkpoint_path, f"mappo_step_{timesteps}.pt")
                )

        return dict(history)

    def evaluate(
        self,
        num_episodes: int = 10,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate trained policy.

        Args:
            num_episodes: Number of evaluation episodes
            deterministic: Use deterministic actions

        Returns:
            Evaluation metrics
        """
        rewards = []
        lengths = []
        successes = []

        for _ in range(num_episodes):
            observations, _ = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False

            while not done:
                actions, _, _ = self.mappo.select_actions(
                    observations, deterministic=deterministic
                )

                next_obs, step_rewards, terminations, truncations, infos = self.env.step(
                    actions
                )

                episode_reward += np.mean(list(step_rewards.values()))
                episode_length += 1

                dones = {
                    aid: terminations.get(aid, False) or truncations.get(aid, False)
                    for aid in observations.keys()
                }
                done = all(dones.values()) or len(next_obs) == 0

                # Check success
                if done:
                    success = any(
                        info.get("intercepted", False)
                        for info in infos.values()
                    )
                    successes.append(float(success))

                observations = next_obs

            rewards.append(episode_reward)
            lengths.append(episode_length)

        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_length": np.mean(lengths),
            "success_rate": np.mean(successes) if successes else 0.0,
        }
