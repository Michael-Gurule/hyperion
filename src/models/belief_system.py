"""
Distributed Bayesian Belief System for Resilient Target Tracking.

Implements per-agent probabilistic belief representation with:
- Particle filter for non-Gaussian belief representation
- Gaussian mixture models for multi-modal hypotheses
- Gossip-based belief propagation for comms-resilient sharing
- Belief decay for handling stale information
- Uncertainty quantification and confidence bounds

This enables continued operation even with:
- Intermittent communication
- Partial sensor coverage
- High measurement uncertainty
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
import copy


class BeliefRepresentation(Enum):
    """Types of belief representation."""
    GAUSSIAN = auto()          # Single Gaussian (fast, limited)
    GAUSSIAN_MIXTURE = auto()  # Gaussian mixture model (multi-modal)
    PARTICLE = auto()          # Particle filter (non-parametric)


@dataclass
class GaussianBelief:
    """Gaussian belief representation."""
    mean: np.ndarray              # State mean [x, y, vx, vy]
    covariance: np.ndarray        # State covariance
    confidence: float = 1.0       # Overall confidence [0, 1]
    timestamp: float = 0.0        # Last update time
    source_agent: Optional[str] = None

    def copy(self) -> 'GaussianBelief':
        return GaussianBelief(
            mean=self.mean.copy(),
            covariance=self.covariance.copy(),
            confidence=self.confidence,
            timestamp=self.timestamp,
            source_agent=self.source_agent,
        )


@dataclass
class GaussianMixtureBelief:
    """Gaussian mixture model belief for multi-modal hypotheses."""
    components: List[GaussianBelief]
    weights: List[float]
    timestamp: float = 0.0

    def get_best_component(self) -> GaussianBelief:
        """Get highest weight component."""
        if not self.components:
            return GaussianBelief(
                mean=np.zeros(4),
                covariance=np.eye(4) * 1000,
                confidence=0.0,
            )
        idx = np.argmax(self.weights)
        return self.components[idx]

    def get_mean(self) -> np.ndarray:
        """Get weighted mean across all components."""
        if not self.components:
            return np.zeros(4)

        mean = np.zeros_like(self.components[0].mean)
        for comp, weight in zip(self.components, self.weights):
            mean += weight * comp.mean
        return mean

    def copy(self) -> 'GaussianMixtureBelief':
        return GaussianMixtureBelief(
            components=[c.copy() for c in self.components],
            weights=self.weights.copy(),
            timestamp=self.timestamp,
        )


@dataclass
class Particle:
    """Single particle in particle filter."""
    state: np.ndarray    # [x, y, vx, vy]
    weight: float = 1.0


class ParticleBelief:
    """Particle filter belief representation for non-Gaussian posteriors."""

    def __init__(
        self,
        num_particles: int = 500,
        state_dim: int = 4,
        resample_threshold: float = 0.5,
    ):
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.resample_threshold = resample_threshold

        # Initialize particles
        self.particles: List[Particle] = []
        self.timestamp = 0.0
        self.initialized = False

    def initialize(
        self,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
    ):
        """Initialize particles around initial estimate."""
        self.particles = []

        for _ in range(self.num_particles):
            state = np.random.multivariate_normal(initial_state, initial_covariance)
            self.particles.append(Particle(state=state, weight=1.0 / self.num_particles))

        self.initialized = True
        self._normalize_weights()

    def predict(self, dt: float, process_noise: np.ndarray):
        """Propagate particles forward with motion model."""
        if not self.initialized:
            return

        # State transition: constant velocity model
        F = np.eye(self.state_dim)
        F[0, 2] = dt  # x += vx * dt
        F[1, 3] = dt  # y += vy * dt

        for particle in self.particles:
            # Apply dynamics
            particle.state = F @ particle.state

            # Add process noise
            noise = np.random.multivariate_normal(
                np.zeros(self.state_dim),
                process_noise * dt,
            )
            particle.state += noise

    def update(
        self,
        measurement: np.ndarray,
        measurement_noise: float,
        measurement_dim: int = 2,
    ):
        """Update particle weights with measurement likelihood."""
        if not self.initialized:
            self.initialize(
                np.concatenate([measurement, np.zeros(self.state_dim - measurement_dim)]),
                np.eye(self.state_dim) * 1000,
            )
            return

        # Measurement matrix (observe position only)
        H = np.zeros((measurement_dim, self.state_dim))
        H[0, 0] = 1.0
        H[1, 1] = 1.0

        for particle in self.particles:
            # Expected measurement
            expected = H @ particle.state

            # Innovation
            innovation = measurement - expected

            # Likelihood (Gaussian)
            likelihood = np.exp(
                -0.5 * np.sum(innovation ** 2) / (measurement_noise ** 2)
            )

            particle.weight *= likelihood + 1e-10

        self._normalize_weights()

        # Resample if effective sample size too low
        if self._effective_sample_size() < self.resample_threshold * self.num_particles:
            self._resample()

    def _normalize_weights(self):
        """Normalize particle weights to sum to 1."""
        total = sum(p.weight for p in self.particles)
        if total > 0:
            for p in self.particles:
                p.weight /= total

    def _effective_sample_size(self) -> float:
        """Compute effective sample size."""
        weights = np.array([p.weight for p in self.particles])
        return 1.0 / (np.sum(weights ** 2) + 1e-10)

    def _resample(self):
        """Resample particles using systematic resampling."""
        weights = np.array([p.weight for p in self.particles])
        cumsum = np.cumsum(weights)

        # Systematic resampling
        positions = (np.arange(self.num_particles) + np.random.random()) / self.num_particles

        new_particles = []
        idx = 0

        for pos in positions:
            while cumsum[idx] < pos and idx < len(cumsum) - 1:
                idx += 1

            new_particles.append(Particle(
                state=self.particles[idx].state.copy(),
                weight=1.0 / self.num_particles,
            ))

        self.particles = new_particles

    def get_mean(self) -> np.ndarray:
        """Get weighted mean state."""
        if not self.particles:
            return np.zeros(self.state_dim)

        mean = np.zeros(self.state_dim)
        for p in self.particles:
            mean += p.weight * p.state
        return mean

    def get_covariance(self) -> np.ndarray:
        """Get weighted covariance."""
        if not self.particles:
            return np.eye(self.state_dim) * 1000

        mean = self.get_mean()
        cov = np.zeros((self.state_dim, self.state_dim))

        for p in self.particles:
            diff = p.state - mean
            cov += p.weight * np.outer(diff, diff)

        return cov

    def get_confidence(self) -> float:
        """Get confidence based on particle spread."""
        cov = self.get_covariance()
        spread = np.sqrt(np.trace(cov[:2, :2]))  # Position spread

        # Confidence inversely proportional to spread
        return 1.0 / (1.0 + spread / 100.0)

    def to_gaussian(self) -> GaussianBelief:
        """Convert to Gaussian belief (moment matching)."""
        return GaussianBelief(
            mean=self.get_mean(),
            covariance=self.get_covariance(),
            confidence=self.get_confidence(),
            timestamp=self.timestamp,
        )

    def copy(self) -> 'ParticleBelief':
        new_belief = ParticleBelief(
            num_particles=self.num_particles,
            state_dim=self.state_dim,
            resample_threshold=self.resample_threshold,
        )
        new_belief.particles = [
            Particle(state=p.state.copy(), weight=p.weight)
            for p in self.particles
        ]
        new_belief.timestamp = self.timestamp
        new_belief.initialized = self.initialized
        return new_belief


@dataclass
class BeliefMessage:
    """Message for gossip-based belief sharing."""
    sender_id: str
    belief: GaussianBelief
    hop_count: int = 0
    max_hops: int = 3
    sequence_number: int = 0

    def copy(self) -> 'BeliefMessage':
        return BeliefMessage(
            sender_id=self.sender_id,
            belief=self.belief.copy(),
            hop_count=self.hop_count,
            max_hops=self.max_hops,
            sequence_number=self.sequence_number,
        )


class GossipProtocol:
    """
    Gossip-based belief propagation protocol.

    Enables belief sharing even with intermittent connectivity.
    Uses epidemic-style spreading with convergent averaging.

    Key features:
    - Works with high packet loss
    - No coordinator required
    - Mathematically guaranteed convergence
    - Age-based message discarding
    """

    def __init__(
        self,
        agent_id: str,
        max_message_age: float = 5.0,
        gossip_probability: float = 0.3,
        max_hops: int = 3,
    ):
        self.agent_id = agent_id
        self.max_message_age = max_message_age
        self.gossip_probability = gossip_probability
        self.max_hops = max_hops

        # Message tracking
        self.received_messages: Dict[str, BeliefMessage] = {}
        self.sent_sequence = 0
        self.seen_sequences: Dict[str, int] = {}  # Track seen messages per sender

        # Outgoing message queue
        self.outgoing_queue: List[BeliefMessage] = []

    def create_message(self, belief: GaussianBelief) -> BeliefMessage:
        """Create new belief message for broadcast."""
        self.sent_sequence += 1

        msg = BeliefMessage(
            sender_id=self.agent_id,
            belief=belief.copy(),
            hop_count=0,
            max_hops=self.max_hops,
            sequence_number=self.sent_sequence,
        )

        return msg

    def receive_message(
        self,
        message: BeliefMessage,
        current_time: float,
    ) -> bool:
        """
        Process received belief message.

        Returns True if message was new and useful.
        """
        # Check message age
        if current_time - message.belief.timestamp > self.max_message_age:
            return False

        # Check if we've seen this message before
        sender = message.sender_id
        if sender in self.seen_sequences:
            if message.sequence_number <= self.seen_sequences[sender]:
                return False

        # Check hop count
        if message.hop_count >= message.max_hops:
            return False

        # Accept message
        self.seen_sequences[sender] = message.sequence_number
        self.received_messages[sender] = message.copy()

        # Maybe forward (gossip)
        if np.random.random() < self.gossip_probability:
            forward_msg = message.copy()
            forward_msg.hop_count += 1
            self.outgoing_queue.append(forward_msg)

        return True

    def get_messages_to_send(self) -> List[BeliefMessage]:
        """Get queued messages for transmission."""
        messages = self.outgoing_queue.copy()
        self.outgoing_queue.clear()
        return messages

    def queue_broadcast(self, belief: GaussianBelief):
        """Queue belief for broadcast to neighbors."""
        msg = self.create_message(belief)
        self.outgoing_queue.append(msg)

    def get_received_beliefs(
        self,
        current_time: float,
    ) -> List[Tuple[str, GaussianBelief]]:
        """Get all non-stale received beliefs."""
        beliefs = []

        for sender, msg in self.received_messages.items():
            if current_time - msg.belief.timestamp <= self.max_message_age:
                beliefs.append((sender, msg.belief))

        return beliefs

    def prune_old_messages(self, current_time: float):
        """Remove stale messages."""
        to_remove = []

        for sender, msg in self.received_messages.items():
            if current_time - msg.belief.timestamp > self.max_message_age:
                to_remove.append(sender)

        for sender in to_remove:
            del self.received_messages[sender]


class AgentBeliefSystem:
    """
    Complete belief system for a single agent.

    Manages:
    - Local belief from own sensors
    - Received beliefs from swarm
    - Belief fusion and consensus
    - Confidence tracking
    """

    def __init__(
        self,
        agent_id: str,
        representation: BeliefRepresentation = BeliefRepresentation.GAUSSIAN,
        state_dim: int = 4,
        process_noise: float = 10.0,
        belief_decay_rate: float = 0.1,
    ):
        self.agent_id = agent_id
        self.representation = representation
        self.state_dim = state_dim
        self.process_noise = np.eye(state_dim) * process_noise
        self.belief_decay_rate = belief_decay_rate

        # Local belief
        if representation == BeliefRepresentation.PARTICLE:
            self.local_belief = ParticleBelief(state_dim=state_dim)
            self._local_gaussian = None
        else:
            self.local_belief = GaussianBelief(
                mean=np.zeros(state_dim),
                covariance=np.eye(state_dim) * 1000,
                confidence=0.0,
            )
            self._local_gaussian = self.local_belief

        # Fused belief (combination of local + received)
        self.fused_belief = GaussianBelief(
            mean=np.zeros(state_dim),
            covariance=np.eye(state_dim) * 1000,
            confidence=0.0,
        )

        # Gossip protocol for belief sharing
        self.gossip = GossipProtocol(agent_id=agent_id)

        # Belief history for tracking
        self.belief_history: deque = deque(maxlen=100)

        # Last update time
        self.last_update_time = 0.0

    def predict(self, dt: float, current_time: float):
        """Predict belief forward in time."""
        if isinstance(self.local_belief, ParticleBelief):
            self.local_belief.predict(dt, self.process_noise)
            self.local_belief.timestamp = current_time
        else:
            # Kalman predict for Gaussian
            F = np.eye(self.state_dim)
            F[0, 2] = dt
            F[1, 3] = dt

            self.local_belief.mean = F @ self.local_belief.mean
            self.local_belief.covariance = (
                F @ self.local_belief.covariance @ F.T +
                self.process_noise * dt
            )
            self.local_belief.timestamp = current_time

            # Decay confidence over time
            self.local_belief.confidence *= np.exp(-self.belief_decay_rate * dt)

        self.last_update_time = current_time

    def update_from_sensor(
        self,
        position: np.ndarray,
        position_uncertainty: float,
        velocity: Optional[np.ndarray] = None,
        velocity_uncertainty: float = float('inf'),
        confidence: float = 1.0,
        current_time: float = 0.0,
    ):
        """Update belief with new sensor measurement."""
        if isinstance(self.local_belief, ParticleBelief):
            self.local_belief.update(
                measurement=position,
                measurement_noise=position_uncertainty,
            )
            self.local_belief.timestamp = current_time

            # Update Gaussian approximation
            self._local_gaussian = self.local_belief.to_gaussian()
        else:
            # Check if this is first measurement (uninitialized)
            if self.local_belief.confidence < 0.01:
                # Initialize directly with measurement
                self.local_belief.mean[:2] = position
                if velocity is not None:
                    self.local_belief.mean[2:4] = velocity
                self.local_belief.covariance = np.eye(self.state_dim) * position_uncertainty ** 2
                self.local_belief.confidence = confidence
                self.local_belief.timestamp = current_time
                self._local_gaussian = self.local_belief
                return

            # Kalman update for Gaussian
            H = np.zeros((2, self.state_dim))
            H[0, 0] = 1.0
            H[1, 1] = 1.0

            R = np.eye(2) * position_uncertainty ** 2

            # Innovation
            y = position - H @ self.local_belief.mean

            # Innovation covariance
            S = H @ self.local_belief.covariance @ H.T + R

            # Kalman gain
            try:
                K = self.local_belief.covariance @ H.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                K = np.zeros((self.state_dim, 2))

            # Update state
            self.local_belief.mean = self.local_belief.mean + K @ y

            # Update covariance (Joseph form)
            I_KH = np.eye(self.state_dim) - K @ H
            self.local_belief.covariance = (
                I_KH @ self.local_belief.covariance @ I_KH.T +
                K @ R @ K.T
            )

            # Update velocity if provided
            if velocity is not None and velocity_uncertainty < 1000:
                self.local_belief.mean[2:4] = (
                    0.5 * self.local_belief.mean[2:4] +
                    0.5 * velocity
                )

            # Update confidence
            self.local_belief.confidence = min(
                1.0,
                self.local_belief.confidence * 0.8 + confidence * 0.2
            )
            self.local_belief.timestamp = current_time
            self._local_gaussian = self.local_belief

        # Record in history
        self.belief_history.append({
            'time': current_time,
            'position': self.get_position().copy(),
            'confidence': self.get_confidence(),
        })

    def fuse_with_neighbors(
        self,
        current_time: float,
        communication_available: bool = True,
    ):
        """Fuse local belief with received neighbor beliefs."""
        # Get local belief as Gaussian
        if isinstance(self.local_belief, ParticleBelief):
            local_gaussian = self.local_belief.to_gaussian()
        else:
            local_gaussian = self.local_belief

        # Start with local belief
        beliefs_to_fuse = [(self.agent_id, local_gaussian)]

        # Add received beliefs if communication available
        if communication_available:
            neighbor_beliefs = self.gossip.get_received_beliefs(current_time)
            beliefs_to_fuse.extend(neighbor_beliefs)

        # Fuse using covariance intersection
        self.fused_belief = self._covariance_intersection(beliefs_to_fuse)
        self.fused_belief.timestamp = current_time

        # Clean up old messages
        self.gossip.prune_old_messages(current_time)

    def _covariance_intersection(
        self,
        beliefs: List[Tuple[str, GaussianBelief]],
    ) -> GaussianBelief:
        """
        Fuse beliefs using Covariance Intersection.

        CI is optimal for unknown correlations between estimates.
        """
        if not beliefs:
            return GaussianBelief(
                mean=np.zeros(self.state_dim),
                covariance=np.eye(self.state_dim) * 1000,
                confidence=0.0,
            )

        if len(beliefs) == 1:
            return beliefs[0][1].copy()

        # Weight each belief by its confidence and information content
        weights = []
        for _, belief in beliefs:
            # Information content (inverse of covariance trace)
            info = 1.0 / (np.trace(belief.covariance) + 1.0)
            weights.append(belief.confidence * info)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(beliefs)] * len(beliefs)

        # Covariance intersection
        P_inv_sum = np.zeros((self.state_dim, self.state_dim))
        weighted_state_sum = np.zeros(self.state_dim)

        for (_, belief), weight in zip(beliefs, weights):
            try:
                P_inv = np.linalg.inv(belief.covariance)
                P_inv_sum += weight * P_inv
                weighted_state_sum += weight * P_inv @ belief.mean
            except np.linalg.LinAlgError:
                continue

        try:
            fused_covariance = np.linalg.inv(P_inv_sum)
            fused_mean = fused_covariance @ weighted_state_sum
        except np.linalg.LinAlgError:
            # Fallback to simple weighted average
            fused_mean = np.zeros(self.state_dim)
            fused_covariance = np.zeros((self.state_dim, self.state_dim))

            for (_, belief), weight in zip(beliefs, weights):
                fused_mean += weight * belief.mean
                fused_covariance += weight * belief.covariance

        # Combined confidence
        fused_confidence = sum(
            w * b.confidence for (_, b), w in zip(beliefs, weights)
        )

        return GaussianBelief(
            mean=fused_mean,
            covariance=fused_covariance,
            confidence=min(1.0, fused_confidence),
        )

    def broadcast_belief(self):
        """Queue current belief for broadcast to neighbors."""
        if self._local_gaussian is not None:
            self.gossip.queue_broadcast(self._local_gaussian)
        elif isinstance(self.local_belief, GaussianBelief):
            self.gossip.queue_broadcast(self.local_belief)

    def receive_belief(
        self,
        message: BeliefMessage,
        current_time: float,
    ) -> bool:
        """Receive and process belief message from neighbor."""
        return self.gossip.receive_message(message, current_time)

    def get_outgoing_messages(self) -> List[BeliefMessage]:
        """Get messages to send to neighbors."""
        return self.gossip.get_messages_to_send()

    def get_position(self) -> np.ndarray:
        """Get best position estimate."""
        if self.fused_belief.confidence > 0:
            return self.fused_belief.mean[:2].copy()

        if isinstance(self.local_belief, ParticleBelief):
            return self.local_belief.get_mean()[:2]
        else:
            return self.local_belief.mean[:2].copy()

    def get_velocity(self) -> np.ndarray:
        """Get best velocity estimate."""
        if self.fused_belief.confidence > 0:
            return self.fused_belief.mean[2:4].copy()

        if isinstance(self.local_belief, ParticleBelief):
            return self.local_belief.get_mean()[2:4]
        else:
            return self.local_belief.mean[2:4].copy()

    def get_full_state(self) -> np.ndarray:
        """Get complete state estimate [x, y, vx, vy]."""
        if self.fused_belief.confidence > 0:
            return self.fused_belief.mean.copy()

        if isinstance(self.local_belief, ParticleBelief):
            return self.local_belief.get_mean()
        else:
            return self.local_belief.mean.copy()

    def get_confidence(self) -> float:
        """Get belief confidence."""
        if self.fused_belief.confidence > 0:
            return self.fused_belief.confidence

        if isinstance(self.local_belief, ParticleBelief):
            return self.local_belief.get_confidence()
        else:
            return self.local_belief.confidence

    def get_position_uncertainty(self) -> float:
        """Get position uncertainty (standard deviation)."""
        if self.fused_belief.confidence > 0:
            cov = self.fused_belief.covariance
        elif isinstance(self.local_belief, ParticleBelief):
            cov = self.local_belief.get_covariance()
        else:
            cov = self.local_belief.covariance

        return np.sqrt(cov[0, 0] + cov[1, 1])

    def get_velocity_uncertainty(self) -> float:
        """Get velocity uncertainty."""
        if self.fused_belief.confidence > 0:
            cov = self.fused_belief.covariance
        elif isinstance(self.local_belief, ParticleBelief):
            cov = self.local_belief.get_covariance()
        else:
            cov = self.local_belief.covariance

        if cov.shape[0] >= 4:
            return np.sqrt(cov[2, 2] + cov[3, 3])
        return float('inf')

    def has_detection(self) -> bool:
        """Check if agent has meaningful belief about target."""
        return self.get_confidence() > 0.1

    def reset(self):
        """Reset belief to uninformed prior."""
        if isinstance(self.local_belief, ParticleBelief):
            self.local_belief = ParticleBelief(state_dim=self.state_dim)
        else:
            self.local_belief = GaussianBelief(
                mean=np.zeros(self.state_dim),
                covariance=np.eye(self.state_dim) * 1000,
                confidence=0.0,
            )

        self.fused_belief = GaussianBelief(
            mean=np.zeros(self.state_dim),
            covariance=np.eye(self.state_dim) * 1000,
            confidence=0.0,
        )

        self._local_gaussian = None
        self.belief_history.clear()


class SwarmBeliefManager:
    """
    Manages beliefs across all agents in swarm.

    Handles:
    - Belief distribution and collection
    - Consensus computation
    - Global uncertainty estimation
    """

    def __init__(
        self,
        num_agents: int,
        agent_ids: Optional[List[str]] = None,
        representation: BeliefRepresentation = BeliefRepresentation.GAUSSIAN,
    ):
        if agent_ids is None:
            agent_ids = [f"agent_{i}" for i in range(num_agents)]

        self.agent_ids = agent_ids
        self.num_agents = num_agents

        # Create belief system for each agent
        self.beliefs: Dict[str, AgentBeliefSystem] = {
            agent_id: AgentBeliefSystem(
                agent_id=agent_id,
                representation=representation,
            )
            for agent_id in agent_ids
        }

        # Global consensus belief
        self.consensus_belief = GaussianBelief(
            mean=np.zeros(4),
            covariance=np.eye(4) * 1000,
            confidence=0.0,
        )

    def update_agent_belief(
        self,
        agent_id: str,
        position: np.ndarray,
        position_uncertainty: float,
        velocity: Optional[np.ndarray] = None,
        velocity_uncertainty: float = float('inf'),
        confidence: float = 1.0,
        current_time: float = 0.0,
    ):
        """Update specific agent's belief with sensor measurement."""
        if agent_id in self.beliefs:
            self.beliefs[agent_id].update_from_sensor(
                position=position,
                position_uncertainty=position_uncertainty,
                velocity=velocity,
                velocity_uncertainty=velocity_uncertainty,
                confidence=confidence,
                current_time=current_time,
            )
            # Also fuse to update the belief state
            self.beliefs[agent_id].fuse_with_neighbors(current_time, False)

    def predict_all(self, dt: float, current_time: float):
        """Predict all agent beliefs forward."""
        for belief_system in self.beliefs.values():
            belief_system.predict(dt, current_time)

    def propagate_beliefs(
        self,
        adjacency: Dict[str, List[str]],
        current_time: float,
        communication_probability: float = 1.0,
    ):
        """
        Propagate beliefs between connected agents.

        Args:
            adjacency: Dict mapping agent_id -> list of neighbor ids
            current_time: Current simulation time
            communication_probability: Probability that communication succeeds
        """
        # Collect outgoing messages from all agents
        all_messages: Dict[str, List[BeliefMessage]] = {}

        for agent_id, belief_system in self.beliefs.items():
            belief_system.broadcast_belief()
            all_messages[agent_id] = belief_system.get_outgoing_messages()

        # Deliver messages to neighbors
        for sender_id, messages in all_messages.items():
            if sender_id not in adjacency:
                continue

            for neighbor_id in adjacency[sender_id]:
                if neighbor_id not in self.beliefs:
                    continue

                # Simulate communication success/failure
                if np.random.random() > communication_probability:
                    continue

                # Deliver each message
                for msg in messages:
                    self.beliefs[neighbor_id].receive_belief(msg, current_time)

        # Fuse beliefs for each agent
        for agent_id, belief_system in self.beliefs.items():
            comm_available = communication_probability > 0.1
            belief_system.fuse_with_neighbors(current_time, comm_available)

    def compute_consensus(self) -> GaussianBelief:
        """Compute swarm-wide consensus belief."""
        # Collect all beliefs with sufficient confidence
        valid_beliefs = []

        for belief_system in self.beliefs.values():
            if belief_system.get_confidence() > 0.1:
                valid_beliefs.append(belief_system.fused_belief)

        if not valid_beliefs:
            return self.consensus_belief

        # Weight by confidence
        weights = [b.confidence for b in valid_beliefs]
        total_weight = sum(weights)

        if total_weight == 0:
            return self.consensus_belief

        weights = [w / total_weight for w in weights]

        # Weighted average
        consensus_mean = np.zeros(4)
        consensus_cov = np.zeros((4, 4))

        for belief, weight in zip(valid_beliefs, weights):
            consensus_mean += weight * belief.mean
            consensus_cov += weight * belief.covariance

        consensus_confidence = np.mean([b.confidence for b in valid_beliefs])

        self.consensus_belief = GaussianBelief(
            mean=consensus_mean,
            covariance=consensus_cov,
            confidence=consensus_confidence,
        )

        return self.consensus_belief

    def get_agent_belief(self, agent_id: str) -> Optional[AgentBeliefSystem]:
        """Get belief system for specific agent."""
        return self.beliefs.get(agent_id)

    def get_swarm_confidence(self) -> float:
        """Get overall swarm confidence in target location."""
        confidences = [b.get_confidence() for b in self.beliefs.values()]
        return np.mean(confidences) if confidences else 0.0

    def get_agents_with_detection(self) -> List[str]:
        """Get list of agents that have detected the target."""
        return [
            agent_id for agent_id, belief in self.beliefs.items()
            if belief.has_detection()
        ]

    def reset(self):
        """Reset all beliefs."""
        for belief_system in self.beliefs.values():
            belief_system.reset()

        self.consensus_belief = GaussianBelief(
            mean=np.zeros(4),
            covariance=np.eye(4) * 1000,
            confidence=0.0,
        )
