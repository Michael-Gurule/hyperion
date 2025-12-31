"""
ARSHI: Autonomous Resilient Sensing & Hive Intelligence

Integrates all resilient sensing components into a unified system:
- OpportunisticSensorSuite: Multi-modal sensing for degraded environments
- BeliefSystem: Distributed Bayesian belief with gossip propagation
- ResilientGNN: Degradation-aware communication layer
- SwarmProprioception: Behavioral inference when blind

Key Capabilities:
1. Graceful degradation from full capability to isolated operation
2. Automatic mode switching based on sensor/comm availability
3. Emergent intelligence through behavioral observation
4. Resilient consensus without centralized coordination

Operating Modes:
- FULL: All sensors and comms operational
- DEGRADED: Reduced sensor suite, intermittent comms
- MINIMAL: Critical sensors only, gossip-based comms
- PROPRIOCEPTIVE: No sensors, infer from swarm behavior
- ISOLATED: No comms, local sensors + memory only
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum, auto
import warnings

from .opportunistic_sensors import (
    OpportunisticSensorSuite,
    SensorReading,
    SignalType,
    SensorStatus,
)
from .belief_system import (
    AgentBeliefSystem,
    SwarmBeliefManager,
    GaussianBelief,
    BeliefMessage,
    BeliefRepresentation,
)
from .resilient_gnn import (
    CommunicationState,
    CommunicationMode,
)


class OperatingMode(Enum):
    """ARSHI operating modes based on available capabilities."""
    FULL = auto()           # All systems nominal
    DEGRADED = auto()       # Some sensors/comms degraded
    MINIMAL = auto()        # Minimum viable operation
    PROPRIOCEPTIVE = auto() # Infer from swarm behavior only
    ISOLATED = auto()       # Completely isolated, local only


@dataclass
class ARSHIState:
    """Current state of ARSHI system for an agent."""
    mode: OperatingMode = OperatingMode.FULL

    # Sensor status
    active_sensors: List[SignalType] = field(default_factory=list)
    sensor_health: Dict[str, float] = field(default_factory=dict)

    # Communication status
    comm_state: CommunicationState = field(default_factory=CommunicationState)
    connected_neighbors: int = 0
    message_success_rate: float = 1.0

    # Belief status
    belief_confidence: float = 0.0
    time_since_detection: float = float('inf')

    # Proprioception status
    proprioception_active: bool = False
    proprioception_confidence: float = 0.0

    # Mode transition history
    mode_history: List[Tuple[float, OperatingMode]] = field(default_factory=list)


@dataclass
class ARSHIObservation:
    """
    Enhanced observation from ARSHI system.

    Extends standard observation with:
    - Multi-sensor fusion results
    - Belief state
    - Uncertainty quantification
    - Operating mode
    """
    # Position and velocity estimates
    target_position: Optional[np.ndarray] = None
    target_velocity: Optional[np.ndarray] = None

    # Uncertainty bounds
    position_uncertainty: float = float('inf')
    velocity_uncertainty: float = float('inf')

    # Confidence and reliability
    detection_confidence: float = 0.0
    sensor_diversity: int = 0  # Number of sensors contributing

    # Operating context
    mode: OperatingMode = OperatingMode.FULL
    jamming_detected: bool = False
    comms_available: bool = True

    # Raw sensor contributions (for explainability)
    sensor_contributions: Dict[str, float] = field(default_factory=dict)

    # Swarm context
    swarm_consensus_position: Optional[np.ndarray] = None
    swarm_consensus_confidence: float = 0.0
    neighbors_with_detection: int = 0


class ARSHIAgent:
    """
    Complete ARSHI system for a single agent.

    Manages:
    - Multi-modal sensing
    - Belief maintenance and fusion
    - Communication with neighbors
    - Mode switching logic
    """

    def __init__(
        self,
        agent_id: str,
        enable_all_sensors: bool = True,
        belief_representation: BeliefRepresentation = BeliefRepresentation.GAUSSIAN,
    ):
        self.agent_id = agent_id

        # Initialize sensor suite
        self.sensors = OpportunisticSensorSuite(
            enable_plasma=enable_all_sensors,
            enable_acoustic=enable_all_sensors,
            enable_thermal=enable_all_sensors,
            enable_magnetic=enable_all_sensors,
            enable_pcl=enable_all_sensors,
            enable_proprioception=enable_all_sensors,
        )

        # Initialize belief system
        self.belief = AgentBeliefSystem(
            agent_id=agent_id,
            representation=belief_representation,
        )

        # Current state
        self.state = ARSHIState()

        # Configuration
        self.mode_switch_hysteresis = 1.0  # seconds
        self.last_mode_switch = 0.0
        self.detection_timeout = 5.0  # seconds without detection -> reduce confidence

        # History for proprioception
        self.neighbor_state_history: Dict[str, List[Dict]] = {}

    def set_environment_state(
        self,
        rf_jamming: float = 0.0,
        gps_denied: bool = False,
        comms_degradation: float = 0.0,
    ):
        """Set current electromagnetic environment."""
        self.sensors.set_jamming_state(
            rf_jamming=rf_jamming,
            gps_jamming=1.0 if gps_denied else 0.0,
            comms_degradation=comms_degradation,
        )

        # Update communication state
        self.state.comm_state.jamming_detected = rf_jamming > 0.3
        self.state.comm_state.jamming_intensity = rf_jamming
        self.state.comm_state.link_quality = 1.0 - comms_degradation * 0.8
        self.state.comm_state.packet_loss_rate = comms_degradation * 0.5

        # Set communication mode
        if comms_degradation > 0.9:
            self.state.comm_state.mode = CommunicationMode.ISOLATED
        elif rf_jamming > 0.7:
            self.state.comm_state.mode = CommunicationMode.JAMMED
        elif comms_degradation > 0.5:
            self.state.comm_state.mode = CommunicationMode.MINIMAL
        elif comms_degradation > 0.2 or rf_jamming > 0.3:
            self.state.comm_state.mode = CommunicationMode.DEGRADED
        else:
            self.state.comm_state.mode = CommunicationMode.FULL

    def update(
        self,
        own_position: np.ndarray,
        own_velocity: np.ndarray,
        target_state: Optional[Dict] = None,
        neighbor_states: Optional[Dict[str, Dict]] = None,
        current_time: float = 0.0,
        dt: float = 0.1,
    ) -> ARSHIObservation:
        """
        Main update loop for ARSHI system.

        Args:
            own_position: Agent's current position
            own_velocity: Agent's current velocity
            target_state: Ground truth target state (for simulation)
            neighbor_states: Dict of neighbor_id -> {position, velocity}
            current_time: Current simulation time
            dt: Time step

        Returns:
            Enhanced observation with all ARSHI data
        """
        # Update neighbor history for proprioception
        if neighbor_states:
            for nid, state in neighbor_states.items():
                if nid not in self.neighbor_state_history:
                    self.neighbor_state_history[nid] = []
                self.neighbor_state_history[nid].append({
                    'time': current_time,
                    'position': state['position'].copy(),
                    'velocity': state['velocity'].copy(),
                })
                # Keep recent history only
                self.neighbor_state_history[nid] = [
                    h for h in self.neighbor_state_history[nid]
                    if current_time - h['time'] < 3.0
                ]

        # Step 1: Predict belief forward
        self.belief.predict(dt, current_time)

        # Step 2: Get readings from all available sensors
        environment = {
            'time': current_time,
            'neighbor_states': neighbor_states or {},
            'neighbor_positions': [s['position'] for s in (neighbor_states or {}).values()],
        }

        sensor_readings = self.sensors.get_all_readings(
            own_position=own_position,
            own_velocity=own_velocity,
            target_state=target_state,
            environment=environment,
        )

        # Step 3: Fuse sensor readings into belief
        if sensor_readings:
            best_reading = self._fuse_sensor_readings(sensor_readings, current_time)
            self.state.time_since_detection = 0.0
        else:
            best_reading = None
            self.state.time_since_detection += dt

        # Step 4: Update operating mode
        self._update_mode(current_time)

        # Step 5: Attempt proprioceptive sensing if in degraded mode
        proprioception_result = None
        if self.state.mode in [OperatingMode.PROPRIOCEPTIVE, OperatingMode.DEGRADED]:
            proprioception_result = self._proprioceptive_sensing(
                own_position, current_time, environment
            )

        # Step 6: Build observation
        observation = self._build_observation(
            best_reading,
            proprioception_result,
            sensor_readings,
        )

        # Update state
        self.state.active_sensors = self.sensors.get_operational_sensors()
        self.state.sensor_health = {
            st: health['reliability']
            for st, health in self.sensors.get_sensor_status().items()
        }
        self.state.belief_confidence = self.belief.get_confidence()
        self.state.connected_neighbors = len(neighbor_states) if neighbor_states else 0

        return observation

    def _fuse_sensor_readings(
        self,
        readings: List[SensorReading],
        current_time: float,
    ) -> SensorReading:
        """Fuse multiple sensor readings into belief."""
        # Weight readings by confidence
        weighted_position = np.zeros(2)
        weighted_velocity = np.zeros(2)
        total_pos_weight = 0.0
        total_vel_weight = 0.0

        for reading in readings:
            if reading.position is not None:
                weight = reading.confidence / (reading.position_uncertainty + 1.0)
                weighted_position += weight * reading.position
                total_pos_weight += weight

                # Update belief
                self.belief.update_from_sensor(
                    position=reading.position,
                    position_uncertainty=reading.position_uncertainty,
                    velocity=reading.velocity,
                    velocity_uncertainty=reading.velocity_uncertainty,
                    confidence=reading.confidence,
                    current_time=current_time,
                )

                # Update sensor reliability
                self.sensors.update_reliability(
                    reading.signal_type,
                    measurement_accepted=True,
                )

            if reading.velocity is not None:
                vel_weight = reading.confidence / (reading.velocity_uncertainty + 1.0)
                weighted_velocity += vel_weight * reading.velocity
                total_vel_weight += vel_weight

        # Compute fused estimates
        if total_pos_weight > 0:
            fused_position = weighted_position / total_pos_weight
        else:
            fused_position = None

        if total_vel_weight > 0:
            fused_velocity = weighted_velocity / total_vel_weight
        else:
            fused_velocity = None

        # Return best reading (highest confidence)
        best = max(readings, key=lambda r: r.confidence)

        return SensorReading(
            signal_type=best.signal_type,
            timestamp=current_time,
            position=fused_position if fused_position is not None else best.position,
            position_uncertainty=min(r.position_uncertainty for r in readings),
            velocity=fused_velocity if fused_velocity is not None else best.velocity,
            velocity_uncertainty=min(
                r.velocity_uncertainty for r in readings
                if r.velocity is not None
            ) if any(r.velocity is not None for r in readings) else float('inf'),
            signal_strength=np.mean([r.signal_strength for r in readings]),
            confidence=np.mean([r.confidence for r in readings]),
            metadata={'num_sensors': len(readings)},
        )

    def _proprioceptive_sensing(
        self,
        own_position: np.ndarray,
        current_time: float,
        environment: Dict,
    ) -> Optional[Dict]:
        """
        Infer target location from neighbor behavior.

        Returns inference result if successful.
        """
        if not self.neighbor_state_history:
            return None

        # Compute acceleration for each neighbor
        accelerations = []
        positions = []

        for nid, history in self.neighbor_state_history.items():
            if len(history) < 2:
                continue

            recent = history[-1]
            older = history[-2]

            dt = recent['time'] - older['time']
            if dt < 0.05:
                continue

            accel = (recent['velocity'] - older['velocity']) / dt
            accel_mag = np.linalg.norm(accel)

            # Significant acceleration indicates maneuvering
            if accel_mag > 3.0:  # m/s^2
                accelerations.append(accel)
                positions.append(recent['position'])

        if len(accelerations) < 2:
            return None

        # Find convergence point
        convergence_point = self._find_acceleration_convergence(
            positions, accelerations
        )

        if convergence_point is None:
            return None

        # Compute confidence based on agreement
        confidence = self._compute_convergence_confidence(
            positions, accelerations, convergence_point
        )

        if confidence < 0.2:
            return None

        self.state.proprioception_active = True
        self.state.proprioception_confidence = confidence

        return {
            'position': convergence_point,
            'confidence': confidence,
            'num_neighbors': len(accelerations),
        }

    def _find_acceleration_convergence(
        self,
        positions: List[np.ndarray],
        accelerations: List[np.ndarray],
    ) -> Optional[np.ndarray]:
        """Find point where acceleration vectors converge."""
        if len(positions) < 2:
            return None

        # Build linear system for intersection
        A = []
        b = []

        for pos, accel in zip(positions, accelerations):
            accel_norm = accel / (np.linalg.norm(accel) + 1e-6)
            perp = np.array([-accel_norm[1], accel_norm[0]])

            A.append(perp)
            b.append(np.dot(perp, pos))

        A = np.array(A)
        b = np.array(b)

        try:
            result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            return result
        except np.linalg.LinAlgError:
            return np.mean(positions, axis=0)

    def _compute_convergence_confidence(
        self,
        positions: List[np.ndarray],
        accelerations: List[np.ndarray],
        target: np.ndarray,
    ) -> float:
        """Compute how well accelerations point toward target."""
        if not positions:
            return 0.0

        alignments = []

        for pos, accel in zip(positions, accelerations):
            to_target = target - pos
            to_target_norm = to_target / (np.linalg.norm(to_target) + 1e-6)
            accel_norm = accel / (np.linalg.norm(accel) + 1e-6)

            alignment = np.dot(accel_norm, to_target_norm)
            alignments.append(max(0, alignment))

        return np.mean(alignments) * min(1.0, len(positions) / 4.0)

    def _update_mode(self, current_time: float):
        """Update operating mode based on current capabilities."""
        # Prevent rapid mode switching
        if current_time - self.last_mode_switch < self.mode_switch_hysteresis:
            return

        # Count operational capabilities
        num_sensors = len(self.sensors.get_operational_sensors())
        comm_quality = self.state.comm_state.link_quality
        has_detection = self.belief.get_confidence() > 0.1
        time_since_detection = self.state.time_since_detection

        # Determine mode
        old_mode = self.state.mode

        if num_sensors >= 4 and comm_quality > 0.7 and has_detection:
            new_mode = OperatingMode.FULL
        elif num_sensors >= 2 and comm_quality > 0.3:
            new_mode = OperatingMode.DEGRADED
        elif num_sensors >= 1 or comm_quality > 0.1:
            new_mode = OperatingMode.MINIMAL
        elif self.state.proprioception_active:
            new_mode = OperatingMode.PROPRIOCEPTIVE
        else:
            new_mode = OperatingMode.ISOLATED

        if new_mode != old_mode:
            self.state.mode = new_mode
            self.last_mode_switch = current_time
            self.state.mode_history.append((current_time, new_mode))

    def _build_observation(
        self,
        best_reading: Optional[SensorReading],
        proprioception: Optional[Dict],
        all_readings: List[SensorReading],
    ) -> ARSHIObservation:
        """Build ARSHI observation from all sources."""
        obs = ARSHIObservation()

        # Use belief as primary source
        if self.belief.has_detection():
            obs.target_position = self.belief.get_position()
            obs.target_velocity = self.belief.get_velocity()
            obs.position_uncertainty = self.belief.get_position_uncertainty()
            obs.velocity_uncertainty = self.belief.get_velocity_uncertainty()
            obs.detection_confidence = self.belief.get_confidence()

        # Supplement with direct sensor reading
        if best_reading is not None:
            if obs.target_position is None:
                obs.target_position = best_reading.position
                obs.target_velocity = best_reading.velocity
                obs.position_uncertainty = best_reading.position_uncertainty
                obs.velocity_uncertainty = best_reading.velocity_uncertainty

            obs.detection_confidence = max(
                obs.detection_confidence,
                best_reading.confidence
            )

        # Supplement with proprioception
        if proprioception is not None:
            if obs.target_position is None:
                obs.target_position = proprioception['position']
                obs.position_uncertainty = 500.0  # High uncertainty
                obs.detection_confidence = proprioception['confidence'] * 0.7

        # Sensor contributions
        obs.sensor_diversity = len(all_readings)
        for reading in all_readings:
            obs.sensor_contributions[reading.signal_type.value] = reading.confidence

        # Operating context
        obs.mode = self.state.mode
        obs.jamming_detected = self.state.comm_state.jamming_detected
        obs.comms_available = self.state.comm_state.mode != CommunicationMode.ISOLATED

        return obs

    def receive_belief_message(
        self,
        message: BeliefMessage,
        current_time: float,
    ) -> bool:
        """Receive and process belief message from neighbor."""
        return self.belief.receive_belief(message, current_time)

    def get_outgoing_messages(self) -> List[BeliefMessage]:
        """Get belief messages to broadcast."""
        self.belief.broadcast_belief()
        return self.belief.get_outgoing_messages()

    def get_state(self) -> ARSHIState:
        """Get current ARSHI state."""
        return self.state

    def get_observation_vector(self) -> np.ndarray:
        """
        Get observation as fixed-size vector for RL policy.

        Returns 15-dim vector:
        - [0:2] target position (normalized)
        - [2:4] target velocity (normalized)
        - [4] position uncertainty (normalized)
        - [5] velocity uncertainty (normalized)
        - [6] detection confidence
        - [7] sensor diversity (normalized)
        - [8] operating mode (one-hot encoded as single value)
        - [9] jamming detected
        - [10] comms available
        - [11] proprioception active
        - [12] belief confidence
        - [13] time since detection (normalized)
        - [14] num connected neighbors (normalized)
        """
        obs = np.zeros(15)

        # Target state
        if self.belief.has_detection():
            pos = self.belief.get_position()
            vel = self.belief.get_velocity()
            obs[0:2] = pos / 10000.0  # Normalize by arena size
            obs[2:4] = vel / 2000.0   # Normalize by max speed
            obs[4] = min(1.0, self.belief.get_position_uncertainty() / 1000.0)
            obs[5] = min(1.0, self.belief.get_velocity_uncertainty() / 500.0)
            obs[6] = self.belief.get_confidence()

        # Sensor info
        obs[7] = len(self.state.active_sensors) / 6.0

        # Mode (0=full, 1=degraded, 2=minimal, 3=proprioceptive, 4=isolated)
        obs[8] = (self.state.mode.value - 1) / 4.0

        # Status flags
        obs[9] = 1.0 if self.state.comm_state.jamming_detected else 0.0
        obs[10] = 1.0 if self.state.comm_state.mode != CommunicationMode.ISOLATED else 0.0
        obs[11] = 1.0 if self.state.proprioception_active else 0.0

        # Belief status
        obs[12] = self.state.belief_confidence
        obs[13] = min(1.0, self.state.time_since_detection / 10.0)
        obs[14] = min(1.0, self.state.connected_neighbors / 10.0)

        return obs

    def reset(self):
        """Reset ARSHI to initial state."""
        self.sensors.reset()
        self.belief.reset()
        self.state = ARSHIState()
        self.neighbor_state_history.clear()


class ARSHISwarm:
    """
    ARSHI system for complete swarm.

    Manages:
    - Per-agent ARSHI instances
    - Belief propagation across swarm
    - Swarm-level consensus
    - Global mode assessment
    """

    def __init__(
        self,
        num_agents: int,
        agent_ids: Optional[List[str]] = None,
    ):
        if agent_ids is None:
            agent_ids = [f"agent_{i}" for i in range(num_agents)]

        self.agent_ids = agent_ids
        self.num_agents = num_agents

        # Create ARSHI agent for each
        self.agents: Dict[str, ARSHIAgent] = {
            agent_id: ARSHIAgent(agent_id)
            for agent_id in agent_ids
        }

        # Swarm-level belief manager
        self.belief_manager = SwarmBeliefManager(
            num_agents=num_agents,
            agent_ids=agent_ids,
        )

    def update(
        self,
        agent_positions: Dict[str, np.ndarray],
        agent_velocities: Dict[str, np.ndarray],
        target_state: Optional[Dict] = None,
        communication_range: float = 1500.0,
        current_time: float = 0.0,
        dt: float = 0.1,
        rf_jamming: float = 0.0,
        comms_degradation: float = 0.0,
    ) -> Dict[str, ARSHIObservation]:
        """
        Update all ARSHI agents.

        Returns observations for all agents.
        """
        observations = {}

        # Build neighbor adjacency based on communication range
        adjacency = self._build_adjacency(
            agent_positions, communication_range
        )

        # Update each agent
        for agent_id in self.agent_ids:
            agent = self.agents[agent_id]

            # Set environment state
            agent.set_environment_state(
                rf_jamming=rf_jamming,
                comms_degradation=comms_degradation,
            )

            # Get neighbor states
            neighbor_ids = adjacency.get(agent_id, [])
            neighbor_states = {
                nid: {
                    'position': agent_positions[nid],
                    'velocity': agent_velocities[nid],
                }
                for nid in neighbor_ids
                if nid in agent_positions
            }

            # Update agent
            obs = agent.update(
                own_position=agent_positions[agent_id],
                own_velocity=agent_velocities[agent_id],
                target_state=target_state,
                neighbor_states=neighbor_states,
                current_time=current_time,
                dt=dt,
            )

            observations[agent_id] = obs

        # Propagate beliefs between agents
        self._propagate_beliefs(adjacency, current_time, comms_degradation)

        # Update swarm consensus
        for agent_id in self.agent_ids:
            obs = observations[agent_id]

            # Add swarm context
            consensus = self.belief_manager.compute_consensus()
            obs.swarm_consensus_position = consensus.mean[:2]
            obs.swarm_consensus_confidence = consensus.confidence

            agents_with_det = self.belief_manager.get_agents_with_detection()
            obs.neighbors_with_detection = len(agents_with_det)

        return observations

    def _build_adjacency(
        self,
        positions: Dict[str, np.ndarray],
        comm_range: float,
    ) -> Dict[str, List[str]]:
        """Build communication adjacency graph."""
        adjacency = {agent_id: [] for agent_id in self.agent_ids}

        for i, id_i in enumerate(self.agent_ids):
            for id_j in self.agent_ids[i+1:]:
                if id_i not in positions or id_j not in positions:
                    continue

                dist = np.linalg.norm(positions[id_i] - positions[id_j])

                if dist < comm_range:
                    adjacency[id_i].append(id_j)
                    adjacency[id_j].append(id_i)

        return adjacency

    def _propagate_beliefs(
        self,
        adjacency: Dict[str, List[str]],
        current_time: float,
        comms_degradation: float,
    ):
        """Propagate belief messages between connected agents."""
        # Collect messages from all agents
        all_messages: Dict[str, List[BeliefMessage]] = {}

        for agent_id, agent in self.agents.items():
            all_messages[agent_id] = agent.get_outgoing_messages()

        # Deliver messages based on adjacency and success probability
        comm_success = 1.0 - comms_degradation * 0.8

        for sender_id, messages in all_messages.items():
            for neighbor_id in adjacency.get(sender_id, []):
                if np.random.random() > comm_success:
                    continue  # Message lost

                for msg in messages:
                    self.agents[neighbor_id].receive_belief_message(
                        msg, current_time
                    )

        # Fuse beliefs for each agent
        for agent in self.agents.values():
            agent.belief.fuse_with_neighbors(
                current_time,
                communication_available=(comms_degradation < 0.9),
            )

    def get_swarm_observation_matrix(self) -> np.ndarray:
        """
        Get observation vectors for all agents as matrix.

        Returns [num_agents, 15] matrix.
        """
        observations = []

        for agent_id in self.agent_ids:
            obs_vec = self.agents[agent_id].get_observation_vector()
            observations.append(obs_vec)

        return np.array(observations)

    def get_swarm_state_summary(self) -> Dict[str, Any]:
        """Get summary of swarm ARSHI state."""
        modes = [a.state.mode for a in self.agents.values()]
        confidences = [a.state.belief_confidence for a in self.agents.values()]

        return {
            'agents_in_full_mode': sum(1 for m in modes if m == OperatingMode.FULL),
            'agents_in_degraded_mode': sum(1 for m in modes if m == OperatingMode.DEGRADED),
            'agents_in_minimal_mode': sum(1 for m in modes if m == OperatingMode.MINIMAL),
            'agents_in_proprioceptive_mode': sum(1 for m in modes if m == OperatingMode.PROPRIOCEPTIVE),
            'agents_isolated': sum(1 for m in modes if m == OperatingMode.ISOLATED),
            'mean_confidence': np.mean(confidences),
            'agents_with_detection': sum(1 for c in confidences if c > 0.1),
            'swarm_consensus_confidence': self.belief_manager.get_swarm_confidence(),
        }

    def reset(self):
        """Reset all agents."""
        for agent in self.agents.values():
            agent.reset()

        self.belief_manager.reset()
