"""
Role-Based Hierarchical Policies using Options Framework.

Implements role-specific policies for swarm coordination:
- SCOUT: Maximize detection coverage, stay distant
- TRACKER: Maintain LOS, medium distance, relay info
- INTERCEPTOR: Close in, optimal firing positions
- SUPPORT: Backup interceptors, fill gaps

The RoleBasedOptionCritic uses:
- A RoleAssigner (manager) that updates roles every k steps
- Per-role action policies (workers)
- GNN embeddings feed into both manager and workers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import IntEnum


class RoleType(IntEnum):
    """Role types matching AgentRole from environment."""
    SCOUT = 0
    TRACKER = 1
    INTERCEPTOR = 2
    SUPPORT = 3


@dataclass
class HierarchicalPolicyConfig:
    """Configuration for hierarchical policy."""
    # Architecture
    embed_dim: int = 64
    hidden_dim: int = 128
    num_roles: int = 4

    # Role assigner (manager)
    role_update_interval: int = 10  # Update roles every k steps
    manager_lr_scale: float = 0.5  # Slower learning rate for manager

    # Policy settings
    action_dim: int = 3  # [thrust, heading, fire]
    use_role_conditioning: bool = True
    share_critic: bool = True

    # Exploration
    role_entropy_bonus: float = 0.01
    policy_entropy_bonus: float = 0.01


class RolePolicy(nn.Module):
    """
    Individual policy network for a specific role.

    Each role has its own policy that specializes in role-specific behavior:
    - SCOUT: Prioritize spread and detection coverage
    - TRACKER: Maintain tracking geometry at medium range
    - INTERCEPTOR: Aggressive approach and firing
    - SUPPORT: Backup positioning and gap filling
    """

    def __init__(
        self,
        role: RoleType,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        action_dim: int = 3,
    ):
        """
        Initialize role-specific policy.

        Args:
            role: The role this policy is for
            embed_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            action_dim: Action space dimension
        """
        super().__init__()

        self.role = role
        self.action_dim = action_dim

        # Role-specific behavior bias (learned)
        self.role_bias = nn.Parameter(torch.zeros(hidden_dim))

        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Action mean and log_std heads
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        # Initialize with role-specific biases
        self._init_role_specific()

    def _init_role_specific(self):
        """Initialize role-specific biases."""
        # These biases encourage role-appropriate behavior from the start
        if self.role == RoleType.SCOUT:
            # Scouts: lower thrust, wider heading range
            nn.init.constant_(self.mean_head.bias[0], -0.2)  # Lower thrust
        elif self.role == RoleType.TRACKER:
            # Trackers: medium thrust, moderate heading
            nn.init.constant_(self.mean_head.bias[0], 0.3)
        elif self.role == RoleType.INTERCEPTOR:
            # Interceptors: high thrust, aggressive approach
            nn.init.constant_(self.mean_head.bias[0], 0.5)  # High thrust
            if self.action_dim > 2:
                nn.init.constant_(self.mean_head.bias[2], 0.3)  # Fire bias
        elif self.role == RoleType.SUPPORT:
            # Support: moderate thrust, flexible
            nn.init.constant_(self.mean_head.bias[0], 0.2)

    def forward(self, embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through role policy.

        Args:
            embedding: Agent embedding from GNN [batch, embed_dim]

        Returns:
            action_mean: Mean action [batch, action_dim]
            action_log_std: Log std of action [batch, action_dim]
        """
        # Apply policy network with role bias
        hidden = self.policy(embedding)
        hidden = hidden + self.role_bias  # Add role-specific bias

        # Compute action distribution
        action_mean = self.mean_head(hidden)
        action_log_std = self.log_std_head(hidden)
        action_log_std = torch.clamp(action_log_std, -5, 2)

        return action_mean, action_log_std


class RoleAssigner(nn.Module):
    """
    Manager network that assigns roles to agents.

    Observes global swarm state and target info to make role assignments.
    Roles can change dynamically based on situation.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_roles: int = 4,
    ):
        """
        Initialize role assigner.

        Args:
            embed_dim: Agent embedding dimension
            hidden_dim: Hidden layer dimension
            num_roles: Number of roles to assign
        """
        super().__init__()

        self.num_roles = num_roles

        # Global context encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Role assignment network (takes agent + global context)
        self.role_assigner = nn.Sequential(
            nn.Linear(embed_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_roles),
        )

        # Role balance predictor (for soft constraints on role distribution)
        self.balance_predictor = nn.Sequential(
            nn.Linear(hidden_dim, num_roles),
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        agent_embeddings: torch.Tensor,
        return_distribution: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Assign roles to agents.

        Args:
            agent_embeddings: Agent embeddings [num_agents, embed_dim]
            return_distribution: Whether to return target role distribution

        Returns:
            role_probs: Role probabilities [num_agents, num_roles]
            role_assignments: Hard role assignments [num_agents]
            target_distribution: Target role distribution [num_roles] (optional)
        """
        num_agents = agent_embeddings.size(0)

        # Compute global context
        global_context = agent_embeddings.mean(dim=0, keepdim=True)
        global_encoded = self.global_encoder(global_context)
        global_encoded = global_encoded.expand(num_agents, -1)

        # Combine agent and global context
        combined = torch.cat([agent_embeddings, global_encoded], dim=-1)

        # Compute role logits
        role_logits = self.role_assigner(combined)
        role_probs = F.softmax(role_logits, dim=-1)

        # Hard assignments (argmax, but use Gumbel-softmax for training)
        if self.training:
            # Gumbel-softmax for differentiable sampling
            role_assignments = F.gumbel_softmax(role_logits, tau=1.0, hard=True).argmax(dim=-1)
        else:
            role_assignments = role_probs.argmax(dim=-1)

        # Target role distribution
        target_dist = None
        if return_distribution:
            target_dist = self.balance_predictor(global_encoded[0:1])

        return role_probs, role_assignments, target_dist

    def compute_role_balance_loss(
        self,
        role_probs: torch.Tensor,
        target_distribution: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute loss for maintaining balanced role distribution.

        Args:
            role_probs: Role probabilities [num_agents, num_roles]
            target_distribution: Target role distribution [num_roles]

        Returns:
            balance_loss: KL divergence from target distribution
        """
        # Current role distribution
        current_dist = role_probs.mean(dim=0)

        # Default target: uniform distribution
        if target_distribution is None:
            target_distribution = torch.ones(
                self.num_roles, device=role_probs.device
            ) / self.num_roles

        # KL divergence
        balance_loss = F.kl_div(
            current_dist.log(), target_distribution, reduction='sum'
        )

        return balance_loss


class RoleBasedOptionCritic(nn.Module):
    """
    Full hierarchical policy with role-based options.

    Architecture:
    - RoleAssigner: Updates roles every k steps based on global state
    - RolePolicies[4]: Per-role action policies
    - Shared Critic: Value estimation
    """

    def __init__(
        self,
        obs_dim: int = 79,
        config: Optional[HierarchicalPolicyConfig] = None,
    ):
        """
        Initialize role-based option critic.

        Args:
            obs_dim: Observation dimension
            config: Policy configuration
        """
        super().__init__()

        self.config = config or HierarchicalPolicyConfig()
        self.obs_dim = obs_dim

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.embed_dim),
            nn.ReLU(),
        )

        # Role assigner (manager)
        self.role_assigner = RoleAssigner(
            embed_dim=self.config.embed_dim,
            hidden_dim=self.config.hidden_dim,
            num_roles=self.config.num_roles,
        )

        # Per-role policies (workers)
        self.role_policies = nn.ModuleList([
            RolePolicy(
                role=RoleType(i),
                embed_dim=self.config.embed_dim,
                hidden_dim=self.config.hidden_dim,
                action_dim=self.config.action_dim,
            )
            for i in range(self.config.num_roles)
        ])

        # Shared critic
        self.critic = nn.Sequential(
            nn.Linear(self.config.embed_dim + self.config.num_roles, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim // 2, 1),
        )

        # Step counter for role updates
        self.step_count = 0
        self.cached_roles: Optional[torch.Tensor] = None

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        """Encode observations to embeddings."""
        return self.obs_encoder(observations)

    def assign_roles(
        self,
        embeddings: torch.Tensor,
        force_update: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assign or use cached roles.

        Args:
            embeddings: Agent embeddings [num_agents, embed_dim]
            force_update: Force role reassignment

        Returns:
            role_probs: Role probabilities [num_agents, num_roles]
            role_assignments: Role indices [num_agents]
        """
        should_update = (
            force_update or
            self.cached_roles is None or
            self.step_count % self.config.role_update_interval == 0
        )

        if should_update:
            role_probs, role_assignments, _ = self.role_assigner(
                embeddings, return_distribution=False
            )
            self.cached_roles = role_assignments
        else:
            # Use cached roles, recompute probs
            role_probs, role_assignments, _ = self.role_assigner(
                embeddings, return_distribution=False
            )
            role_assignments = self.cached_roles

        return role_probs, role_assignments

    def get_actions(
        self,
        embeddings: torch.Tensor,
        role_assignments: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get actions from role-specific policies.

        Args:
            embeddings: Agent embeddings [num_agents, embed_dim]
            role_assignments: Role indices [num_agents]
            deterministic: Whether to use mean actions

        Returns:
            actions: Sampled actions [num_agents, action_dim]
            log_probs: Action log probabilities [num_agents]
            entropies: Policy entropies [num_agents]
        """
        num_agents = embeddings.size(0)
        device = embeddings.device

        actions = torch.zeros(num_agents, self.config.action_dim, device=device)
        log_probs = torch.zeros(num_agents, device=device)
        entropies = torch.zeros(num_agents, device=device)

        # Process each role
        for role_idx in range(self.config.num_roles):
            mask = (role_assignments == role_idx)
            if not mask.any():
                continue

            role_embeddings = embeddings[mask]
            role_policy = self.role_policies[role_idx]

            # Get action distribution
            action_mean, action_log_std = role_policy(role_embeddings)
            action_std = torch.exp(action_log_std)

            # Sample or use mean
            if deterministic:
                role_actions = action_mean
                role_log_probs = torch.zeros(role_embeddings.size(0), device=device)
            else:
                dist = torch.distributions.Normal(action_mean, action_std)
                role_actions = dist.rsample()  # Reparameterized sampling
                role_log_probs = dist.log_prob(role_actions).sum(dim=-1)

            role_entropy = torch.distributions.Normal(action_mean, action_std).entropy().sum(dim=-1)

            # Apply action bounds
            role_actions = self._bound_actions(role_actions)

            # Store results
            actions[mask] = role_actions
            log_probs[mask] = role_log_probs
            entropies[mask] = role_entropy

        return actions, log_probs, entropies

    def _bound_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Apply bounds to actions."""
        # Thrust: [0, 1]
        actions[:, 0] = torch.sigmoid(actions[:, 0])
        # Heading: [-1, 1]
        actions[:, 1] = torch.tanh(actions[:, 1])
        # Fire: [0, 1]
        if self.config.action_dim > 2:
            actions[:, 2] = torch.sigmoid(actions[:, 2])
        return actions

    def get_values(
        self,
        embeddings: torch.Tensor,
        role_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get value estimates.

        Args:
            embeddings: Agent embeddings [num_agents, embed_dim]
            role_probs: Role probabilities [num_agents, num_roles]

        Returns:
            values: Value estimates [num_agents]
        """
        critic_input = torch.cat([embeddings, role_probs], dim=-1)
        values = self.critic(critic_input).squeeze(-1)
        return values

    def forward(
        self,
        observations: torch.Tensor,
        deterministic: bool = False,
        force_role_update: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            observations: Agent observations [num_agents, obs_dim]
            deterministic: Use mean actions
            force_role_update: Force role reassignment

        Returns:
            Dictionary with actions, log_probs, values, roles, etc.
        """
        self.step_count += 1

        # Encode observations
        embeddings = self.encode(observations)

        # Assign roles
        role_probs, role_assignments = self.assign_roles(
            embeddings, force_update=force_role_update
        )

        # Get actions from role-specific policies
        actions, log_probs, entropies = self.get_actions(
            embeddings, role_assignments, deterministic
        )

        # Get values
        values = self.get_values(embeddings, role_probs)

        return {
            "actions": actions,
            "log_probs": log_probs,
            "values": values,
            "role_probs": role_probs,
            "role_assignments": role_assignments,
            "entropies": entropies,
            "embeddings": embeddings,
        }

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        role_assignments: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate actions for PPO update.

        Args:
            observations: Agent observations
            actions: Actions taken
            role_assignments: Optional fixed role assignments

        Returns:
            values, log_probs, entropies, role_probs
        """
        embeddings = self.encode(observations)

        # Get roles
        role_probs, computed_roles, _ = self.role_assigner(embeddings)
        if role_assignments is None:
            role_assignments = computed_roles

        # Evaluate actions under each role's policy
        num_agents = embeddings.size(0)
        device = embeddings.device

        log_probs = torch.zeros(num_agents, device=device)
        entropies = torch.zeros(num_agents, device=device)

        for role_idx in range(self.config.num_roles):
            mask = (role_assignments == role_idx)
            if not mask.any():
                continue

            role_embeddings = embeddings[mask]
            role_actions = actions[mask]
            role_policy = self.role_policies[role_idx]

            # Get distribution
            action_mean, action_log_std = role_policy(role_embeddings)
            action_std = torch.exp(action_log_std)

            # Inverse transform actions for log_prob
            unbounded_actions = self._unbound_actions(role_actions)

            dist = torch.distributions.Normal(action_mean, action_std)
            role_log_probs = dist.log_prob(unbounded_actions).sum(dim=-1)
            role_entropy = dist.entropy().sum(dim=-1)

            log_probs[mask] = role_log_probs
            entropies[mask] = role_entropy

        # Get values
        values = self.get_values(embeddings, role_probs)

        return {
            "values": values,
            "log_probs": log_probs,
            "entropies": entropies,
            "role_probs": role_probs,
        }

    def _unbound_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Inverse of action bounding for log_prob computation."""
        unbounded = actions.clone()
        # Inverse sigmoid for thrust and fire
        eps = 1e-6
        unbounded[:, 0] = torch.log(actions[:, 0].clamp(eps, 1-eps) / (1 - actions[:, 0].clamp(eps, 1-eps)))
        # Inverse tanh for heading
        unbounded[:, 1] = torch.atanh(actions[:, 1].clamp(-1+eps, 1-eps))
        if self.config.action_dim > 2:
            unbounded[:, 2] = torch.log(actions[:, 2].clamp(eps, 1-eps) / (1 - actions[:, 2].clamp(eps, 1-eps)))
        return unbounded

    def reset_roles(self):
        """Reset cached roles for new episode."""
        self.cached_roles = None
        self.step_count = 0


class HierarchicalMAPPO:
    """
    MAPPO with hierarchical role-based policies.

    Extends standard MAPPO with:
    - Separate optimizers for manager (role assigner) and workers (role policies)
    - Role-based advantage computation
    - Manager updates every k steps, workers every step
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 3,
        num_agents: int = 50,
        config: Optional[HierarchicalPolicyConfig] = None,
        device: str = "cpu",
    ):
        """
        Initialize Hierarchical MAPPO.

        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            num_agents: Number of agents
            config: Policy configuration
            device: Compute device
        """
        self.config = config or HierarchicalPolicyConfig(action_dim=action_dim)
        self.device = device
        self.num_agents = num_agents

        # Policy network
        self.policy = RoleBasedOptionCritic(
            obs_dim=obs_dim,
            config=self.config,
        ).to(device)

        # Separate optimizers for manager and workers
        manager_params = list(self.policy.role_assigner.parameters())
        worker_params = (
            list(self.policy.obs_encoder.parameters()) +
            list(self.policy.role_policies.parameters()) +
            list(self.policy.critic.parameters())
        )

        self.manager_optimizer = torch.optim.Adam(
            manager_params,
            lr=3e-4 * self.config.manager_lr_scale,
        )
        self.worker_optimizer = torch.optim.Adam(
            worker_params,
            lr=3e-4,
        )

        # Rollout buffer
        self.buffer: List[Dict[str, Any]] = []

    def select_actions(
        self,
        observations: Dict[str, np.ndarray],
        deterministic: bool = False,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, float], Dict[str, int]]:
        """
        Select actions for all agents.

        Returns:
            actions, log_probs, values, roles
        """
        # Stack observations
        agent_ids = list(observations.keys())
        obs_tensor = torch.tensor(
            np.stack([observations[aid] for aid in agent_ids]),
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            outputs = self.policy(obs_tensor, deterministic=deterministic)

        # Convert to dicts
        actions = {
            aid: outputs["actions"][i].cpu().numpy()
            for i, aid in enumerate(agent_ids)
        }
        log_probs = {
            aid: outputs["log_probs"][i].item()
            for i, aid in enumerate(agent_ids)
        }
        values = {
            aid: outputs["values"][i].item()
            for i, aid in enumerate(agent_ids)
        }
        roles = {
            aid: outputs["role_assignments"][i].item()
            for i, aid in enumerate(agent_ids)
        }

        return actions, log_probs, values, roles

    def store_transition(
        self,
        observations: Dict[str, np.ndarray],
        actions: Dict[str, np.ndarray],
        rewards: Dict[str, float],
        dones: Dict[str, bool],
        log_probs: Dict[str, float],
        values: Dict[str, float],
        roles: Dict[str, int],
    ):
        """Store transition in buffer."""
        self.buffer.append({
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "log_probs": log_probs,
            "values": values,
            "roles": roles,
        })

    def update(
        self,
        last_values: Dict[str, float],
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_param: float = 0.2,
        num_epochs: int = 10,
    ) -> Dict[str, float]:
        """
        Update policy using collected rollouts.

        Returns training statistics.
        """
        if len(self.buffer) == 0:
            return {}

        # Compute advantages
        advantages, returns = self._compute_gae(last_values, gamma, gae_lambda)

        # Prepare batch data
        agent_ids = list(self.buffer[0]["observations"].keys())
        batch_obs = []
        batch_actions = []
        batch_old_log_probs = []
        batch_advantages = []
        batch_returns = []
        batch_roles = []

        for t, transition in enumerate(self.buffer):
            for i, aid in enumerate(agent_ids):
                batch_obs.append(transition["observations"][aid])
                batch_actions.append(transition["actions"][aid])
                batch_old_log_probs.append(transition["log_probs"][aid])
                batch_advantages.append(advantages[t][i])
                batch_returns.append(returns[t][i])
                batch_roles.append(transition["roles"][aid])

        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float32, device=self.device)
        batch_actions = torch.tensor(np.array(batch_actions), dtype=torch.float32, device=self.device)
        batch_old_log_probs = torch.tensor(batch_old_log_probs, dtype=torch.float32, device=self.device)
        batch_advantages = torch.tensor(batch_advantages, dtype=torch.float32, device=self.device)
        batch_returns = torch.tensor(batch_returns, dtype=torch.float32, device=self.device)
        batch_roles = torch.tensor(batch_roles, dtype=torch.long, device=self.device)

        # Normalize advantages
        batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

        # Training stats
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_role_entropy = 0.0

        # PPO update epochs
        for epoch in range(num_epochs):
            # Evaluate actions
            eval_outputs = self.policy.evaluate_actions(
                batch_obs, batch_actions, batch_roles
            )

            # Policy loss (PPO clip)
            ratio = torch.exp(eval_outputs["log_probs"] - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(eval_outputs["values"], batch_returns)

            # Entropy bonus
            entropy = eval_outputs["entropies"].mean()
            role_probs_detached = eval_outputs["role_probs"].detach()
            role_entropy = -(role_probs_detached * role_probs_detached.log().clamp(min=-10)).sum(dim=-1).mean()

            # Total loss for workers
            worker_loss = (
                policy_loss +
                0.5 * value_loss -
                self.config.policy_entropy_bonus * entropy
            )

            # Update workers
            self.worker_optimizer.zero_grad()
            worker_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.worker_optimizer.step()

            # Update manager (role assigner) less frequently
            if epoch % 2 == 0:
                # Re-evaluate for manager update with fresh graph
                manager_outputs = self.policy.evaluate_actions(
                    batch_obs, batch_actions, batch_roles
                )

                # Role balance loss
                role_balance_loss = self.policy.role_assigner.compute_role_balance_loss(
                    manager_outputs["role_probs"]
                )

                manager_role_entropy = -(manager_outputs["role_probs"] * manager_outputs["role_probs"].log().clamp(min=-10)).sum(dim=-1).mean()

                manager_loss = (
                    role_balance_loss -
                    self.config.role_entropy_bonus * manager_role_entropy
                )

                self.manager_optimizer.zero_grad()
                manager_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.role_assigner.parameters(), 0.5)
                self.manager_optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            total_role_entropy += role_entropy.item()

        # Clear buffer
        self.buffer = []
        self.policy.reset_roles()

        return {
            "policy_loss": total_policy_loss / num_epochs,
            "value_loss": total_value_loss / num_epochs,
            "entropy": total_entropy / num_epochs,
            "role_entropy": total_role_entropy / num_epochs,
        }

    def _compute_gae(
        self,
        last_values: Dict[str, float],
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """Compute GAE advantages and returns."""
        agent_ids = list(self.buffer[0]["observations"].keys())
        T = len(self.buffer)
        N = len(agent_ids)

        # Initialize
        advantages = [[0.0] * N for _ in range(T)]
        returns = [[0.0] * N for _ in range(T)]

        # Compute backwards
        gae = [0.0] * N

        for t in reversed(range(T)):
            for i, aid in enumerate(agent_ids):
                if t == T - 1:
                    next_value = last_values.get(aid, 0.0)
                    next_done = self.buffer[t]["dones"].get(aid, True)
                else:
                    next_value = self.buffer[t + 1]["values"][aid]
                    next_done = self.buffer[t + 1]["dones"].get(aid, False)

                current_value = self.buffer[t]["values"][aid]
                reward = self.buffer[t]["rewards"].get(aid, 0.0)
                done = self.buffer[t]["dones"].get(aid, False)

                # TD error
                delta = reward + gamma * next_value * (1 - float(next_done)) - current_value

                # GAE
                gae[i] = delta + gamma * gae_lambda * (1 - float(done)) * gae[i]

                advantages[t][i] = gae[i]
                returns[t][i] = gae[i] + current_value

        return advantages, returns

    def save(self, path: str):
        """Save model."""
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "manager_optimizer": self.manager_optimizer.state_dict(),
            "worker_optimizer": self.worker_optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.manager_optimizer.load_state_dict(checkpoint["manager_optimizer"])
        self.worker_optimizer.load_state_dict(checkpoint["worker_optimizer"])
