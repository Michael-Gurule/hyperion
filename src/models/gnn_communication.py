"""
Graph Neural Network communication layer for scalable multi-agent coordination.

Enables dynamic message passing between agents with arbitrary swarm sizes,
supporting 50-100+ agents efficiently through attention-based mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch_geometric.nn import MessagePassing, GATv2Conv, global_mean_pool
from torch_geometric.data import Data, Batch


class AgentNodeEncoder(nn.Module):
    """
    Encodes individual agent observations into node features.
    Handles variable observation dimensions and normalizes features.
    """

    def __init__(
        self,
        obs_dim: int = 25,
        hidden_dim: int = 128,
        embed_dim: int = 64,
    ):
        """
        Initialize agent encoder.

        Args:
            obs_dim: Dimension of raw agent observation
            hidden_dim: Hidden layer dimension
            embed_dim: Output embedding dimension
        """
        super().__init__()

        # Self-state encoder (position, velocity, fuel, detection)
        self.self_encoder = nn.Sequential(
            nn.Linear(10, hidden_dim),  # 5 self + 5 target
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )

        # Full observation backup encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )

        self.embed_dim = embed_dim

    def forward(
        self,
        obs: torch.Tensor,
        use_full_obs: bool = False,
    ) -> torch.Tensor:
        """
        Encode agent observation to embedding.

        Args:
            obs: Agent observation [batch, obs_dim] or [obs_dim]
            use_full_obs: Whether to use full observation encoder

        Returns:
            Agent embedding [batch, embed_dim]
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        if use_full_obs:
            return self.obs_encoder(obs)

        # Extract self-state (first 10 dims: 5 self + 5 target)
        self_state = obs[:, :10]
        return self.self_encoder(self_state)


class GNNCommunicationLayer(MessagePassing):
    """
    Graph Neural Network layer for inter-agent communication.
    Uses attention-based message passing with edge features.
    """

    def __init__(
        self,
        node_feat_dim: int = 64,
        edge_dim: int = 8,
        hidden_dim: int = 128,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initialize GNN communication layer.

        Args:
            node_feat_dim: Node feature dimension
            edge_dim: Edge feature dimension
            hidden_dim: Hidden layer dimension
            heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__(aggr="add", node_dim=0)  # node_dim=0 for PyG MessagePassing

        self.node_feat_dim = node_feat_dim
        self.edge_dim = edge_dim
        self.heads = heads

        # Message computation
        self.message_mlp = nn.Sequential(
            nn.Linear(node_feat_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_feat_dim),
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(node_feat_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, heads),
        )

        # Node update
        self.update_mlp = nn.Sequential(
            nn.Linear(node_feat_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_feat_dim),
        )

        # Layer normalization
        self.norm = nn.LayerNorm(node_feat_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through communication layer.

        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]

        Returns:
            Updated node features [num_nodes, node_dim]
        """
        # Create dummy edge features if not provided
        if edge_attr is None:
            edge_attr = torch.zeros(
                edge_index.size(1), self.edge_dim, device=x.device
            )

        # Propagate messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Residual connection with normalization
        out = self.norm(x + out)

        return out

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute messages from source to target nodes.

        Args:
            x_i: Target node features [num_edges, node_dim]
            x_j: Source node features [num_edges, node_dim]
            edge_attr: Edge features [num_edges, edge_dim]

        Returns:
            Messages [num_edges, node_dim]
        """
        # Concatenate features for message computation
        concat = torch.cat([x_i, x_j, edge_attr], dim=-1)

        # Compute message content
        message = self.message_mlp(concat)

        # Compute attention weights
        attention_logits = self.attention(concat)
        attention_weights = F.softmax(attention_logits, dim=-1)

        # Weight message by attention (average over heads)
        weighted_message = message * attention_weights.mean(dim=-1, keepdim=True)

        return weighted_message

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Update node features with aggregated messages.

        Args:
            aggr_out: Aggregated messages [num_nodes, node_dim]
            x: Original node features [num_nodes, node_dim]

        Returns:
            Updated features [num_nodes, node_dim]
        """
        concat = torch.cat([x, aggr_out], dim=-1)
        return self.update_mlp(concat)


class SwarmGNN(nn.Module):
    """
    Full Graph Neural Network for swarm coordination.
    Supports variable number of agents (50-100+) with efficient message passing.
    """

    def __init__(
        self,
        obs_dim: int = 25,
        hidden_dim: int = 128,
        embed_dim: int = 64,
        num_gnn_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        communication_range: float = 1500.0,
    ):
        """
        Initialize swarm GNN.

        Args:
            obs_dim: Agent observation dimension
            hidden_dim: Hidden layer dimension
            embed_dim: Node embedding dimension
            num_gnn_layers: Number of GNN communication layers
            heads: Attention heads per layer
            dropout: Dropout rate
            communication_range: Max distance for edge creation
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.embed_dim = embed_dim
        self.communication_range = communication_range

        # Node encoder
        self.node_encoder = AgentNodeEncoder(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
        )

        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GNNCommunicationLayer(
                node_feat_dim=embed_dim,
                edge_dim=8,
                hidden_dim=hidden_dim,
                heads=heads,
                dropout=dropout,
            )
            for _ in range(num_gnn_layers)
        ])

        # Alternative: Use GATv2 for more established architecture
        self.use_gatv2 = False
        if self.use_gatv2:
            self.gatv2_layers = nn.ModuleList([
                GATv2Conv(
                    in_channels=embed_dim,
                    out_channels=embed_dim // heads,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=8,
                    concat=True,
                )
                for _ in range(num_gnn_layers)
            ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def build_graph(
        self,
        positions: torch.Tensor,
        max_neighbors: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build communication graph from agent positions.

        Args:
            positions: Agent positions [num_agents, 2]
            max_neighbors: Maximum neighbors per agent (None for all)

        Returns:
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features (distance, angle) [num_edges, 8]
        """
        num_agents = positions.size(0)
        device = positions.device

        # Compute pairwise distances
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # [N, N, 2]
        distances = torch.norm(diff, dim=-1)  # [N, N]

        # Create edges within communication range
        mask = (distances < self.communication_range) & (distances > 0)

        # Optionally limit neighbors
        if max_neighbors is not None:
            # Keep only k nearest neighbors
            topk_mask = torch.zeros_like(mask)
            for i in range(num_agents):
                valid_dist = distances[i].clone()
                valid_dist[~mask[i]] = float('inf')
                _, indices = torch.topk(valid_dist, min(max_neighbors, mask[i].sum()), largest=False)
                topk_mask[i, indices] = True
            mask = mask & topk_mask

        # Get edge indices
        edge_index = mask.nonzero(as_tuple=False).t()  # [2, num_edges]

        if edge_index.size(1) == 0:
            # No edges - return empty
            return edge_index, torch.zeros(0, 8, device=device)

        # Compute edge features
        src, dst = edge_index[0], edge_index[1]
        rel_pos = positions[dst] - positions[src]  # [num_edges, 2]
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)  # [num_edges, 1]

        # Normalized features
        rel_pos_norm = rel_pos / (dist + 1e-6)  # Direction
        dist_norm = dist / self.communication_range  # Normalized distance

        # Angle features
        angle = torch.atan2(rel_pos[:, 1], rel_pos[:, 0]).unsqueeze(-1)
        sin_angle = torch.sin(angle)
        cos_angle = torch.cos(angle)

        # Combine edge features [dist_norm, rel_x, rel_y, cos, sin, dist_sq, zeros]
        edge_attr = torch.cat([
            dist_norm,
            rel_pos_norm,
            cos_angle,
            sin_angle,
            (dist_norm ** 2),
            torch.zeros(edge_index.size(1), 2, device=device),  # Padding
        ], dim=-1)

        return edge_index, edge_attr

    def forward(
        self,
        observations: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through swarm GNN.

        Args:
            observations: Agent observations [num_agents, obs_dim]
            positions: Agent positions [num_agents, 2] (extracted from obs if None)
            batch: Batch assignment for batched graphs [num_agents]

        Returns:
            Enhanced agent embeddings [num_agents, embed_dim]
        """
        # Extract positions from observations if not provided
        if positions is None:
            # Assume positions are first 2 dims (need to denormalize)
            positions = observations[:, :2] * 10000.0  # arena_size

        # Encode observations to node features
        x = self.node_encoder(observations, use_full_obs=True)

        # Build communication graph
        edge_index, edge_attr = self.build_graph(positions)

        # Apply GNN layers
        if self.use_gatv2:
            for layer in self.gatv2_layers:
                x = layer(x, edge_index, edge_attr)
                x = F.relu(x)
        else:
            for layer in self.gnn_layers:
                x = layer(x, edge_index, edge_attr)

        # Output projection
        x = self.output_proj(x)

        return x

    def forward_batched(
        self,
        observations_list: List[torch.Tensor],
        positions_list: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Process multiple environments in batch.

        Args:
            observations_list: List of observations per environment
            positions_list: List of positions per environment

        Returns:
            List of enhanced embeddings per environment
        """
        # Build batched graph
        data_list = []
        for obs, pos in zip(observations_list, positions_list):
            x = self.node_encoder(obs, use_full_obs=True)
            edge_index, edge_attr = self.build_graph(pos)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            data_list.append(data)

        batch = Batch.from_data_list(data_list)

        # Forward through GNN
        x = batch.x
        if self.use_gatv2:
            for layer in self.gatv2_layers:
                x = layer(x, batch.edge_index, batch.edge_attr)
                x = F.relu(x)
        else:
            for layer in self.gnn_layers:
                x = layer(x, batch.edge_index, batch.edge_attr)

        x = self.output_proj(x)

        # Unbatch results
        results = []
        ptr = 0
        for i, obs in enumerate(observations_list):
            n = obs.size(0)
            results.append(x[ptr:ptr + n])
            ptr += n

        return results


class CoordinationHead(nn.Module):
    """
    Coordination head for generating swarm-level decisions.
    Combines individual agent embeddings with global context.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_roles: int = 4,  # e.g., scout, tracker, interceptor, support
    ):
        """
        Initialize coordination head.

        Args:
            embed_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            num_roles: Number of coordination roles
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_roles = num_roles

        # Global context aggregation
        self.global_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )

        # Role assignment
        self.role_classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_roles),
        )

        # Coordination score (how well coordinated)
        self.coordination_scorer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        agent_embeddings: torch.Tensor,
        return_roles: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Generate coordination outputs.

        Args:
            agent_embeddings: Agent embeddings [num_agents, embed_dim]
            return_roles: Whether to return role assignments

        Returns:
            enhanced_embeddings: Embeddings with global context [num_agents, embed_dim]
            role_probs: Role probabilities [num_agents, num_roles] (if return_roles)
            coordination_score: Overall coordination quality [1]
        """
        if agent_embeddings.dim() == 2:
            agent_embeddings = agent_embeddings.unsqueeze(0)  # Add batch dim

        # Self-attention for global context
        attended, _ = self.global_attention(
            agent_embeddings,
            agent_embeddings,
            agent_embeddings,
        )

        # Global context (mean pooling)
        global_context = attended.mean(dim=1, keepdim=True)  # [B, 1, embed_dim]
        global_context = global_context.expand(-1, agent_embeddings.size(1), -1)

        # Enhanced embeddings
        enhanced = attended + agent_embeddings  # Residual

        # Role assignment
        role_probs = None
        if return_roles:
            concat_for_role = torch.cat([enhanced, global_context], dim=-1)
            role_logits = self.role_classifier(concat_for_role)
            role_probs = F.softmax(role_logits, dim=-1)

        # Coordination score
        global_embed = attended.mean(dim=1)  # [B, embed_dim]
        coordination_score = self.coordination_scorer(global_embed)

        # Remove batch dim if single
        if enhanced.size(0) == 1:
            enhanced = enhanced.squeeze(0)
            if role_probs is not None:
                role_probs = role_probs.squeeze(0)
            coordination_score = coordination_score.squeeze(0)

        return enhanced, role_probs, coordination_score


class SwarmCoordinationNetwork(nn.Module):
    """
    Complete swarm coordination network combining GNN and coordination head.
    Designed for 50-100+ agent swarms with efficient processing.
    """

    def __init__(
        self,
        obs_dim: int = 25,
        action_dim: int = 2,
        hidden_dim: int = 128,
        embed_dim: int = 64,
        num_gnn_layers: int = 3,
        num_roles: int = 4,
        communication_range: float = 1500.0,
    ):
        """
        Initialize complete coordination network.

        Args:
            obs_dim: Agent observation dimension
            action_dim: Action space dimension
            hidden_dim: Hidden layer dimension
            embed_dim: Embedding dimension
            num_gnn_layers: Number of GNN layers
            num_roles: Number of coordination roles
            communication_range: Communication range for graph construction
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim

        # GNN for communication
        self.gnn = SwarmGNN(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            num_gnn_layers=num_gnn_layers,
            communication_range=communication_range,
        )

        # Coordination head
        self.coordination = CoordinationHead(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_roles=num_roles,
        )

        # Policy head (actor)
        self.actor = nn.Sequential(
            nn.Linear(embed_dim + num_roles, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim * 2),  # mean + log_std
        )

        # Value head (critic)
        self.critic = nn.Sequential(
            nn.Linear(embed_dim + num_roles, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        observations: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass for policy and value.

        Args:
            observations: Agent observations [num_agents, obs_dim]
            positions: Agent positions [num_agents, 2]

        Returns:
            Dictionary with action_mean, action_log_std, value, roles, coordination_score
        """
        # GNN message passing
        gnn_embeddings = self.gnn(observations, positions)

        # Coordination with role assignment
        enhanced, role_probs, coord_score = self.coordination(
            gnn_embeddings, return_roles=True
        )

        # Concatenate with role info for policy/value
        role_context = role_probs  # [num_agents, num_roles]
        actor_input = torch.cat([enhanced, role_context], dim=-1)

        # Actor output
        actor_out = self.actor(actor_input)
        action_mean = actor_out[:, :self.action_dim]
        action_log_std = actor_out[:, self.action_dim:]
        action_log_std = torch.clamp(action_log_std, -5, 2)

        # Critic output
        value = self.critic(actor_input)

        return {
            "action_mean": action_mean,
            "action_log_std": action_log_std,
            "value": value,
            "roles": role_probs,
            "coordination_score": coord_score,
            "embeddings": enhanced,
        }

    def get_action(
        self,
        observations: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            observations: Agent observations
            positions: Agent positions
            deterministic: Whether to use mean action

        Returns:
            actions: Sampled actions [num_agents, action_dim]
            log_probs: Log probabilities [num_agents]
        """
        outputs = self.forward(observations, positions)

        mean = outputs["action_mean"]
        log_std = outputs["action_log_std"]
        std = torch.exp(log_std)

        if deterministic:
            actions = mean
            log_probs = torch.zeros(mean.size(0), device=mean.device)
        else:
            # Sample from Gaussian
            dist = torch.distributions.Normal(mean, std)
            actions = dist.sample()
            log_probs = dist.log_prob(actions).sum(dim=-1)

        # Clamp actions to valid range
        actions = torch.tanh(actions)  # Scale to [-1, 1]
        actions[:, 0] = (actions[:, 0] + 1) / 2  # Thrust: [0, 1]

        return actions, log_probs

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.

        Args:
            observations: Agent observations
            actions: Actions taken
            positions: Agent positions

        Returns:
            values: Value estimates
            log_probs: Action log probabilities
            entropy: Policy entropy
        """
        outputs = self.forward(observations, positions)

        mean = outputs["action_mean"]
        log_std = outputs["action_log_std"]
        std = torch.exp(log_std)
        value = outputs["value"]

        dist = torch.distributions.Normal(mean, std)

        # Transform actions back for log_prob
        actions_transformed = actions.clone()
        actions_transformed[:, 0] = actions_transformed[:, 0] * 2 - 1  # [0,1] -> [-1,1]

        log_probs = dist.log_prob(actions_transformed).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return value.squeeze(-1), log_probs, entropy
