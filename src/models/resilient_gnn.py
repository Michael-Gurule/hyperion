"""
Degradation-Aware Graph Neural Network for Resilient Swarm Communication.

Extends standard GNN communication with:
- Graceful handling of missing/unreliable edges
- Uncertainty propagation through messages
- Fallback to local information when isolated
- Bandwidth-constrained message passing
- Jamming detection and adaptation

Designed to maintain coordination even under:
- 50%+ communication failure
- Intermittent connectivity
- Active jamming
- Partial swarm isolation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum, auto
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch


class CommunicationMode(Enum):
    """Operating modes for communication system."""
    FULL = auto()           # Full bandwidth, all links available
    DEGRADED = auto()       # Reduced bandwidth, some link failures
    MINIMAL = auto()        # Severe degradation, critical messages only
    ISOLATED = auto()       # No communication, rely on local + proprioception
    JAMMED = auto()         # Active jamming detected


@dataclass
class CommunicationState:
    """Current state of communication system."""
    mode: CommunicationMode = CommunicationMode.FULL
    link_quality: float = 1.0           # [0, 1] overall link quality
    bandwidth_available: float = 1.0     # [0, 1] available bandwidth
    jamming_detected: bool = False
    jamming_intensity: float = 0.0
    packet_loss_rate: float = 0.0
    active_neighbors: int = 0
    message_latency: float = 0.0         # seconds


class UncertaintyEncoder(nn.Module):
    """
    Encodes observation uncertainty alongside features.

    Enables the network to reason about reliability of inputs.
    """

    def __init__(
        self,
        obs_dim: int = 25,
        uncertainty_dim: int = 5,
        embed_dim: int = 64,
    ):
        super().__init__()

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
        )

        self.uncertainty_encoder = nn.Sequential(
            nn.Linear(uncertainty_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim),
            nn.Sigmoid(),  # Output as attention weights
        )

        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        obs: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode observation with uncertainty weighting.

        Args:
            obs: Observations [batch, obs_dim]
            uncertainty: Uncertainty values [batch, uncertainty_dim]
                        Higher values = less reliable

        Returns:
            Uncertainty-weighted embeddings [batch, embed_dim]
        """
        obs_embed = self.obs_encoder(obs)

        if uncertainty is None:
            return obs_embed

        # Uncertainty attention (lower uncertainty = higher weight)
        uncertainty_weights = self.uncertainty_encoder(uncertainty)
        reliability_weights = 1.0 - uncertainty_weights

        # Weight features by reliability
        weighted_obs = obs_embed * reliability_weights

        # Concatenate and fuse
        combined = torch.cat([obs_embed, weighted_obs], dim=-1)
        return self.fusion(combined)


class ResilientMessageLayer(nn.Module):
    """
    Message passing layer that handles unreliable edges.

    Implemented without PyTorch Geometric MessagePassing for better compatibility.

    Features:
    - Edge reliability weighting
    - Message dropout for robustness
    - Fallback aggregation when isolated
    - Bandwidth-aware message compression
    """

    def __init__(
        self,
        node_dim: int = 64,
        edge_dim: int = 8,
        hidden_dim: int = 128,
        heads: int = 4,
        dropout: float = 0.1,
        message_dropout: float = 0.2,
    ):
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.heads = heads
        self.message_dropout = message_dropout

        # Edge reliability prediction
        self.reliability_predictor = nn.Sequential(
            nn.Linear(edge_dim + node_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Message computation
        self.message_net = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim),
        )

        # Attention for message weighting
        self.attention_net = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, heads),
        )

        # Update with self-loop for isolation handling
        self.update_net = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim),
        )

        # Self-information preservation (for when isolated)
        self.self_update = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
        )

        self.norm = nn.LayerNorm(node_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        edge_reliability: Optional[torch.Tensor] = None,
        comm_state: Optional[CommunicationState] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with reliability-aware message passing.

        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Graph edges [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            edge_reliability: Per-edge reliability [num_edges]
            comm_state: Current communication state

        Returns:
            Updated node features [num_nodes, node_dim]
            Edge reliabilities used [num_edges]
        """
        num_nodes = x.size(0)
        num_edges = edge_index.size(1) if edge_index.numel() > 0 else 0

        # Handle no edges case (isolated mode)
        if num_edges == 0:
            # Use self-update only
            self_info = self.self_update(x)
            return self.norm(x + self_info), torch.tensor([], device=x.device)

        # Default edge attributes
        if edge_attr is None:
            edge_attr = torch.zeros(num_edges, self.edge_dim, device=x.device)

        src, dst = edge_index[0], edge_index[1]

        # Compute or use provided reliability
        if edge_reliability is None:
            edge_reliability = self._compute_reliability(
                x, edge_index, edge_attr, comm_state
            )

        # Apply communication state effects
        if comm_state is not None:
            edge_reliability = edge_reliability * comm_state.link_quality

            # Random link failures based on packet loss
            if comm_state.packet_loss_rate > 0 and self.training:
                mask = torch.rand(num_edges, device=x.device) > comm_state.packet_loss_rate
                edge_reliability = edge_reliability * mask.float()

        # Compute messages
        x_src, x_dst = x[src], x[dst]
        concat = torch.cat([x_dst, x_src, edge_attr], dim=-1)

        # Message content
        messages = self.message_net(concat)

        # Attention weights
        attention_logits = self.attention_net(concat)
        attention = F.softmax(attention_logits, dim=-1).mean(dim=-1, keepdim=True)

        # Apply reliability weighting
        reliability_weight = edge_reliability.unsqueeze(-1)

        # Apply message dropout during training
        if self.training and self.message_dropout > 0:
            dropout_mask = torch.rand_like(reliability_weight) > self.message_dropout
            reliability_weight = reliability_weight * dropout_mask.float()

        weighted_messages = messages * attention * reliability_weight

        # Aggregate messages to destination nodes
        aggregated = torch.zeros(num_nodes, self.node_dim, device=x.device)
        aggregated.scatter_add_(0, dst.unsqueeze(-1).expand(-1, self.node_dim), weighted_messages)

        # Self-information for nodes with few neighbors
        self_info = self.self_update(x)

        # Compute neighbor count per node
        neighbor_counts = torch.zeros(num_nodes, device=x.device)
        neighbor_counts.scatter_add_(0, dst, edge_reliability)

        # Blend aggregated and self based on neighbor count
        isolation_factor = torch.exp(-neighbor_counts).unsqueeze(-1)

        # Update with isolation-aware blending
        concat_update = torch.cat([x, aggregated], dim=-1)
        update = self.update_net(concat_update)

        # Blend: more self-info when isolated
        blended = (1 - isolation_factor) * update + isolation_factor * self_info

        return self.norm(x + blended), edge_reliability

    def _compute_reliability(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        comm_state: Optional[CommunicationState],
    ) -> torch.Tensor:
        """Predict reliability for each edge."""
        src, dst = edge_index[0], edge_index[1]
        x_src, x_dst = x[src], x[dst]

        # Concatenate source, destination, edge features
        edge_features = torch.cat([x_src, x_dst, edge_attr], dim=-1)

        reliability = self.reliability_predictor(edge_features).squeeze(-1)

        # Apply communication state modifier
        if comm_state is not None:
            if comm_state.mode == CommunicationMode.JAMMED:
                reliability = reliability * 0.3
            elif comm_state.mode == CommunicationMode.MINIMAL:
                reliability = reliability * 0.5
            elif comm_state.mode == CommunicationMode.DEGRADED:
                reliability = reliability * 0.8

        return reliability


class BandwidthCompressor(nn.Module):
    """
    Compresses messages for bandwidth-constrained communication.

    Uses learned compression to maintain critical information
    while reducing message size.
    """

    def __init__(
        self,
        full_dim: int = 64,
        compressed_dims: List[int] = [32, 16, 8],
    ):
        super().__init__()

        self.full_dim = full_dim
        self.compressed_dims = compressed_dims

        # Encoder for each compression level
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(full_dim, dim * 2),
                nn.ReLU(),
                nn.Linear(dim * 2, dim),
            )
            for dim in compressed_dims
        ])

        # Decoder for each compression level
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.ReLU(),
                nn.Linear(dim * 2, full_dim),
            )
            for dim in compressed_dims
        ])

    def compress(
        self,
        x: torch.Tensor,
        compression_level: int = 0,
    ) -> torch.Tensor:
        """
        Compress message to specified level.

        Args:
            x: Full message [batch, full_dim]
            compression_level: 0=least compression, higher=more

        Returns:
            Compressed message [batch, compressed_dim]
        """
        level = min(compression_level, len(self.encoders) - 1)
        return self.encoders[level](x)

    def decompress(
        self,
        x: torch.Tensor,
        compression_level: int = 0,
    ) -> torch.Tensor:
        """Decompress message back to full dimension."""
        level = min(compression_level, len(self.decoders) - 1)
        return self.decoders[level](x)

    def get_compression_ratio(self, level: int) -> float:
        """Get compression ratio for level."""
        level = min(level, len(self.compressed_dims) - 1)
        return self.compressed_dims[level] / self.full_dim


class ResilientSwarmGNN(nn.Module):
    """
    Complete resilient GNN for swarm coordination under degraded conditions.

    Handles:
    - Unreliable edges with dropout
    - Bandwidth constraints with compression
    - Isolation detection and fallback
    - Jamming detection and adaptation
    """

    def __init__(
        self,
        obs_dim: int = 25,
        uncertainty_dim: int = 5,
        hidden_dim: int = 128,
        embed_dim: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        communication_range: float = 1500.0,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.embed_dim = embed_dim
        self.communication_range = communication_range

        # Uncertainty-aware encoder
        self.encoder = UncertaintyEncoder(
            obs_dim=obs_dim,
            uncertainty_dim=uncertainty_dim,
            embed_dim=embed_dim,
        )

        # Resilient message passing layers
        self.gnn_layers = nn.ModuleList([
            ResilientMessageLayer(
                node_dim=embed_dim,
                edge_dim=8,
                hidden_dim=hidden_dim,
                heads=heads,
                dropout=dropout,
                message_dropout=0.2 if i > 0 else 0.1,  # Increase dropout in deeper layers
            )
            for i in range(num_layers)
        ])

        # Bandwidth compressor
        self.compressor = BandwidthCompressor(
            full_dim=embed_dim,
            compressed_dims=[32, 16, 8],
        )

        # Isolation detector
        self.isolation_detector = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Communication state estimator
        self.comm_state_net = nn.Sequential(
            nn.Linear(embed_dim + 4, hidden_dim),  # +4 for comm features
            nn.ReLU(),
            nn.Linear(hidden_dim, 5),  # [link_quality, bandwidth, jamming, latency, isolation]
        )

    def build_graph(
        self,
        positions: torch.Tensor,
        comm_state: Optional[CommunicationState] = None,
        max_neighbors: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build communication graph with reliability-aware edge selection.
        """
        num_agents = positions.size(0)
        device = positions.device

        # Compute pairwise distances
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)
        distances = torch.norm(diff, dim=-1)

        # Effective communication range (reduced under jamming)
        effective_range = self.communication_range
        if comm_state is not None:
            if comm_state.mode == CommunicationMode.JAMMED:
                effective_range *= 0.5
            elif comm_state.mode == CommunicationMode.DEGRADED:
                effective_range *= 0.8

        # Create edges within range
        mask = (distances < effective_range) & (distances > 0)

        # Limit neighbors for scalability
        if max_neighbors is not None:
            topk_mask = torch.zeros_like(mask)
            for i in range(num_agents):
                valid_dist = distances[i].clone()
                valid_dist[~mask[i]] = float('inf')
                k = min(max_neighbors, mask[i].sum().item())
                if k > 0:
                    _, indices = torch.topk(valid_dist, k, largest=False)
                    topk_mask[i, indices] = True
            mask = mask & topk_mask

        edge_index = mask.nonzero(as_tuple=False).t()

        if edge_index.size(1) == 0:
            return edge_index, torch.zeros(0, 8, device=device)

        # Compute edge features
        src, dst = edge_index
        rel_pos = positions[dst] - positions[src]
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)

        dist_norm = dist / effective_range
        rel_pos_norm = rel_pos / (dist + 1e-6)

        angle = torch.atan2(rel_pos[:, 1], rel_pos[:, 0]).unsqueeze(-1)

        edge_attr = torch.cat([
            dist_norm,
            rel_pos_norm,
            torch.cos(angle),
            torch.sin(angle),
            dist_norm ** 2,
            torch.zeros(edge_index.size(1), 2, device=device),
        ], dim=-1)

        return edge_index, edge_attr

    def forward(
        self,
        observations: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        uncertainties: Optional[torch.Tensor] = None,
        comm_state: Optional[CommunicationState] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with degradation awareness.

        Args:
            observations: Agent observations [num_agents, obs_dim]
            positions: Agent positions [num_agents, 2]
            uncertainties: Observation uncertainties [num_agents, uncertainty_dim]
            comm_state: Current communication state

        Returns:
            Dictionary with embeddings, isolation scores, edge reliabilities
        """
        num_agents = observations.size(0)
        device = observations.device

        # Extract positions if not provided
        if positions is None:
            positions = observations[:, :2] * 10000.0

        # Encode observations with uncertainty
        x = self.encoder(observations, uncertainties)

        # Build communication graph
        edge_index, edge_attr = self.build_graph(positions, comm_state)

        # Track edge reliabilities
        all_edge_reliabilities = []

        # Apply GNN layers
        for layer in self.gnn_layers:
            x, edge_reliability = layer(
                x, edge_index, edge_attr,
                comm_state=comm_state,
            )
            if edge_reliability.numel() > 0:
                all_edge_reliabilities.append(edge_reliability)

        # Compute isolation score per agent
        isolation_scores = self.isolation_detector(x).squeeze(-1)

        # Estimate communication state from learned features
        if comm_state is not None:
            comm_features = torch.tensor([
                comm_state.link_quality,
                comm_state.bandwidth_available,
                comm_state.jamming_intensity,
                comm_state.packet_loss_rate,
            ], device=device).expand(num_agents, -1)
        else:
            comm_features = torch.zeros(num_agents, 4, device=device)

        comm_input = torch.cat([x, comm_features], dim=-1)
        estimated_comm = torch.sigmoid(self.comm_state_net(comm_input))

        # Output projection
        output = self.output_proj(x)

        # Aggregate edge reliabilities
        if all_edge_reliabilities:
            mean_reliability = torch.stack(all_edge_reliabilities).mean(dim=0)
        else:
            mean_reliability = torch.tensor([], device=device)

        return {
            'embeddings': output,
            'isolation_scores': isolation_scores,
            'edge_reliabilities': mean_reliability,
            'estimated_comm_state': estimated_comm,
            'num_active_edges': edge_index.size(1) if edge_index.numel() > 0 else 0,
        }

    def forward_degraded(
        self,
        observations: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        uncertainties: Optional[torch.Tensor] = None,
        degradation_level: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with synthetic degradation for robustness training.

        Args:
            degradation_level: [0, 1] amount of synthetic degradation
        """
        # Create synthetic communication state
        comm_state = CommunicationState(
            mode=CommunicationMode.DEGRADED if degradation_level > 0.3 else CommunicationMode.FULL,
            link_quality=1.0 - degradation_level * 0.7,
            bandwidth_available=1.0 - degradation_level * 0.5,
            jamming_detected=degradation_level > 0.6,
            jamming_intensity=max(0, degradation_level - 0.4),
            packet_loss_rate=degradation_level * 0.4,
        )

        return self.forward(observations, positions, uncertainties, comm_state)


class ResilientCoordinationHead(nn.Module):
    """
    Coordination head that adapts to communication state.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_roles: int = 4,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_roles = num_roles

        # Self-attention for global context (when available)
        self.global_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )

        # Local role assignment (for isolated operation)
        self.local_role_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_roles),
        )

        # Global role assignment (with coordination)
        self.global_role_net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_roles),
        )

        # Coordination quality estimator
        self.coord_quality = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        isolation_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute role assignments with isolation awareness.

        Args:
            embeddings: Agent embeddings [num_agents, embed_dim]
            isolation_scores: Per-agent isolation [num_agents]

        Returns:
            role_probs: Role probabilities [num_agents, num_roles]
            enhanced_embeddings: Context-enhanced embeddings
            coordination_score: Overall coordination quality
        """
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
            isolation_scores = isolation_scores.unsqueeze(0)

        batch_size, num_agents, _ = embeddings.shape

        # Global attention for context
        attended, _ = self.global_attention(
            embeddings, embeddings, embeddings
        )

        global_context = attended.mean(dim=1, keepdim=True).expand(-1, num_agents, -1)

        # Local role assignment
        local_roles = self.local_role_net(embeddings)

        # Global role assignment
        global_input = torch.cat([attended, global_context], dim=-1)
        global_roles = self.global_role_net(global_input)

        # Blend based on isolation
        isolation_weights = isolation_scores.unsqueeze(-1)
        role_logits = isolation_weights * local_roles + (1 - isolation_weights) * global_roles
        role_probs = F.softmax(role_logits, dim=-1)

        # Enhanced embeddings
        enhanced = attended + embeddings  # Residual

        # Coordination quality (lower when more agents isolated)
        mean_embed = embeddings.mean(dim=1)
        coordination_score = self.coord_quality(mean_embed)

        # Reduce coordination score based on isolation
        mean_isolation = isolation_scores.mean(dim=1, keepdim=True)
        coordination_score = coordination_score * (1 - mean_isolation * 0.5)

        # Remove batch dim if single
        if batch_size == 1:
            role_probs = role_probs.squeeze(0)
            enhanced = enhanced.squeeze(0)
            coordination_score = coordination_score.squeeze(0)

        return role_probs, enhanced, coordination_score


class ResilientSwarmCoordinator(nn.Module):
    """
    Complete resilient swarm coordination network.

    Combines degradation-aware GNN with adaptive coordination.
    """

    def __init__(
        self,
        obs_dim: int = 25,
        action_dim: int = 2,
        uncertainty_dim: int = 5,
        hidden_dim: int = 128,
        embed_dim: int = 64,
        num_gnn_layers: int = 3,
        num_roles: int = 4,
        communication_range: float = 1500.0,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim

        # Resilient GNN
        self.gnn = ResilientSwarmGNN(
            obs_dim=obs_dim,
            uncertainty_dim=uncertainty_dim,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            num_layers=num_gnn_layers,
            communication_range=communication_range,
        )

        # Coordination head
        self.coordination = ResilientCoordinationHead(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_roles=num_roles,
        )

        # Policy head (actor)
        self.actor = nn.Sequential(
            nn.Linear(embed_dim + num_roles + 1, hidden_dim),  # +1 for isolation
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim * 2),
        )

        # Value head (critic)
        self.critic = nn.Sequential(
            nn.Linear(embed_dim + num_roles + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        observations: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        uncertainties: Optional[torch.Tensor] = None,
        comm_state: Optional[CommunicationState] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.
        """
        # GNN forward
        gnn_out = self.gnn(observations, positions, uncertainties, comm_state)

        embeddings = gnn_out['embeddings']
        isolation_scores = gnn_out['isolation_scores']

        # Coordination
        role_probs, enhanced, coord_score = self.coordination(
            embeddings, isolation_scores
        )

        # Policy input: embeddings + roles + isolation
        policy_input = torch.cat([
            enhanced,
            role_probs,
            isolation_scores.unsqueeze(-1),
        ], dim=-1)

        # Actor output
        actor_out = self.actor(policy_input)
        action_mean = actor_out[:, :self.action_dim]
        action_log_std = torch.clamp(actor_out[:, self.action_dim:], -5, 2)

        # Critic output
        value = self.critic(policy_input)

        return {
            'action_mean': action_mean,
            'action_log_std': action_log_std,
            'value': value,
            'roles': role_probs,
            'coordination_score': coord_score,
            'embeddings': enhanced,
            'isolation_scores': isolation_scores,
            'edge_reliabilities': gnn_out['edge_reliabilities'],
            'estimated_comm_state': gnn_out['estimated_comm_state'],
        }

    def get_action(
        self,
        observations: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        uncertainties: Optional[torch.Tensor] = None,
        comm_state: Optional[CommunicationState] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        outputs = self.forward(observations, positions, uncertainties, comm_state)

        mean = outputs['action_mean']
        log_std = outputs['action_log_std']
        std = torch.exp(log_std)

        if deterministic:
            actions = mean
            log_probs = torch.zeros(mean.size(0), device=mean.device)
        else:
            dist = torch.distributions.Normal(mean, std)
            actions = dist.sample()
            log_probs = dist.log_prob(actions).sum(dim=-1)

        # Clamp and scale actions
        actions = torch.tanh(actions)
        actions[:, 0] = (actions[:, 0] + 1) / 2  # Thrust: [0, 1]

        return actions, log_probs
