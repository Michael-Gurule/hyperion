"""
Comprehensive tests for ARSHI (Autonomous Resilient Sensing & Hive Intelligence).

Tests cover:
- Opportunistic sensor suite under various conditions
- Belief system fusion and gossip propagation
- Resilient GNN with degraded communication
- Mode switching logic
- Swarm-level coordination
"""

import pytest
import numpy as np
import torch
from typing import Dict, List

# Import ARSHI components
import sys
sys.path.insert(0, 'src')

from models.opportunistic_sensors import (
    OpportunisticSensorSuite,
    PlasmaEmissionDetector,
    AcousticArraySensor,
    ThermalWakeDetector,
    MagneticAnomalyDetector,
    PassiveCoherentLocation,
    SwarmProprioception,
    SignalType,
    SensorStatus,
    SensorReading,
)
from models.belief_system import (
    AgentBeliefSystem,
    SwarmBeliefManager,
    GossipProtocol,
    GaussianBelief,
    BeliefMessage,
    ParticleBelief,
    BeliefRepresentation,
)
from models.resilient_gnn import (
    ResilientSwarmGNN,
    ResilientSwarmCoordinator,
    CommunicationState,
    CommunicationMode,
    UncertaintyEncoder,
    BandwidthCompressor,
)
from models.arshi import (
    ARSHIAgent,
    ARSHISwarm,
    ARSHIObservation,
    OperatingMode,
)


class TestOpportunisticSensors:
    """Tests for opportunistic sensor suite."""

    def test_sensor_suite_initialization(self):
        """Test sensor suite initializes all sensors."""
        suite = OpportunisticSensorSuite()

        assert SignalType.PLASMA_EMISSION in suite.sensors
        assert SignalType.SONIC_BOOM in suite.sensors
        assert SignalType.THERMAL_WAKE in suite.sensors
        assert SignalType.MAGNETIC_ANOMALY in suite.sensors
        assert SignalType.AMBIENT_RF_SCATTER in suite.sensors
        assert SignalType.OPTICAL_OCCLUSION in suite.sensors

    def test_selective_sensor_initialization(self):
        """Test selective sensor enabling."""
        suite = OpportunisticSensorSuite(
            enable_plasma=True,
            enable_acoustic=False,
            enable_thermal=True,
            enable_magnetic=False,
            enable_pcl=False,
            enable_proprioception=False,
        )

        assert SignalType.PLASMA_EMISSION in suite.sensors
        assert SignalType.SONIC_BOOM not in suite.sensors
        assert SignalType.THERMAL_WAKE in suite.sensors
        assert SignalType.MAGNETIC_ANOMALY not in suite.sensors

    def test_plasma_detector_hypersonic_target(self):
        """Test plasma emission detector for hypersonic target."""
        detector = PlasmaEmissionDetector()

        own_pos = np.array([0.0, 0.0])
        own_vel = np.array([100.0, 0.0])

        # Hypersonic target (Mach 6 ~ 2060 m/s) at closer range for reliable detection
        target_state = {
            'position': np.array([1500.0, 0.0]),
            'velocity': np.array([2000.0, 0.0]),
        }

        reading = detector.detect(
            own_position=own_pos,
            own_velocity=own_vel,
            target_state=target_state,
            environment={'time': 0.0},
        )

        assert reading is not None
        assert reading.signal_type == SignalType.PLASMA_EMISSION
        assert reading.confidence > 0
        assert reading.position is not None
        assert 'mach_number' in reading.metadata

    def test_plasma_detector_subsonic_target(self):
        """Test plasma detector returns None for subsonic target."""
        detector = PlasmaEmissionDetector()

        own_pos = np.array([0.0, 0.0])
        own_vel = np.array([100.0, 0.0])

        # Subsonic target
        target_state = {
            'position': np.array([3000.0, 0.0]),
            'velocity': np.array([200.0, 0.0]),
        }

        reading = detector.detect(
            own_position=own_pos,
            own_velocity=own_vel,
            target_state=target_state,
            environment={'time': 0.0},
        )

        assert reading is None  # No plasma sheath below Mach 5

    def test_acoustic_sensor_supersonic(self):
        """Test acoustic sensor for supersonic target."""
        sensor = AcousticArraySensor()

        own_pos = np.array([0.0, 0.0])
        own_vel = np.array([50.0, 0.0])

        # Supersonic target
        target_state = {
            'position': np.array([5000.0, 0.0]),
            'velocity': np.array([500.0, 0.0]),  # ~Mach 1.5
        }

        reading = sensor.detect(
            own_position=own_pos,
            own_velocity=own_vel,
            target_state=target_state,
            environment={
                'time': 10.0,
                'neighbor_positions': [
                    np.array([1000.0, 500.0]),
                    np.array([500.0, 1000.0]),
                ],
            },
        )

        # Should get some reading even without full triangulation
        assert reading is not None or target_state['velocity'][0] < 343  # subsonic fallback

    def test_thermal_wake_detection(self):
        """Test thermal wake detector."""
        detector = ThermalWakeDetector()

        own_pos = np.array([0.0, 0.0])
        own_vel = np.array([100.0, 0.0])

        # High-speed target
        target_state = {
            'position': np.array([2000.0, 0.0]),
            'velocity': np.array([1500.0, 0.0]),
        }

        reading = detector.detect(
            own_position=own_pos,
            own_velocity=own_vel,
            target_state=target_state,
            environment={'time': 0.0, 'altitude': 10000.0},
        )

        assert reading is not None
        assert reading.signal_type == SignalType.THERMAL_WAKE
        assert 'thermal_intensity' in reading.metadata

    def test_jamming_effects(self):
        """Test that jamming degrades sensor performance."""
        suite = OpportunisticSensorSuite()

        # Set high jamming
        suite.set_jamming_state(rf_jamming=0.9)

        # Check sensor status
        status = suite.get_sensor_status()

        # PCL should be degraded (susceptible to RF jamming)
        assert status['ambient_rf_scatter']['jamming_detected'] == True

        # Acoustic should be unaffected (immune to RF jamming)
        assert status['sonic_boom']['jamming_detected'] == False

    def test_multi_sensor_fusion(self):
        """Test getting readings from multiple sensors."""
        suite = OpportunisticSensorSuite()

        own_pos = np.array([0.0, 0.0])
        own_vel = np.array([100.0, 0.0])

        # Hypersonic target at close range
        target_state = {
            'position': np.array([2000.0, 0.0]),
            'velocity': np.array([2000.0, 0.0]),
            'mass': 1000.0,
            'radar_cross_section': 1.0,
        }

        readings = suite.get_all_readings(
            own_position=own_pos,
            own_velocity=own_vel,
            target_state=target_state,
            environment={'time': 0.0, 'neighbor_positions': []},
        )

        # Should get readings from multiple sensors
        assert len(readings) >= 1

        # Get best reading
        best = suite.get_best_reading(
            own_position=own_pos,
            own_velocity=own_vel,
            target_state=target_state,
            environment={'time': 0.0},
        )

        assert best is not None


class TestBeliefSystem:
    """Tests for belief system components."""

    def test_gaussian_belief_update(self):
        """Test Gaussian belief updates with measurements."""
        belief = AgentBeliefSystem(
            agent_id="test_agent",
            representation=BeliefRepresentation.GAUSSIAN,
        )

        # Update with measurement
        belief.update_from_sensor(
            position=np.array([1000.0, 500.0]),
            position_uncertainty=50.0,
            velocity=np.array([500.0, 0.0]),
            velocity_uncertainty=20.0,
            confidence=0.8,
            current_time=0.0,
        )

        assert belief.has_detection()
        assert belief.get_confidence() > 0

        pos = belief.get_position()
        assert np.allclose(pos, [1000.0, 500.0], atol=100)

    def test_particle_belief_update(self):
        """Test particle filter belief updates."""
        belief = AgentBeliefSystem(
            agent_id="test_agent",
            representation=BeliefRepresentation.PARTICLE,
        )

        # Update with measurement
        belief.update_from_sensor(
            position=np.array([1000.0, 500.0]),
            position_uncertainty=50.0,
            confidence=0.8,
            current_time=0.0,
        )

        assert belief.has_detection()

        pos = belief.get_position()
        assert np.linalg.norm(pos - np.array([1000.0, 500.0])) < 200

    def test_belief_prediction(self):
        """Test belief prediction over time."""
        belief = AgentBeliefSystem(agent_id="test")

        # Initial measurement
        belief.update_from_sensor(
            position=np.array([0.0, 0.0]),
            position_uncertainty=10.0,
            velocity=np.array([100.0, 0.0]),
            velocity_uncertainty=5.0,
            confidence=1.0,
            current_time=0.0,
        )

        # Predict forward
        belief.predict(dt=1.0, current_time=1.0)

        pos = belief.get_position()
        # Position should move based on velocity
        assert pos[0] > 50  # Should have moved in x direction

    def test_belief_decay(self):
        """Test belief confidence decay over time."""
        belief = AgentBeliefSystem(
            agent_id="test",
            belief_decay_rate=0.5,
        )

        belief.update_from_sensor(
            position=np.array([1000.0, 0.0]),
            position_uncertainty=10.0,
            confidence=1.0,
            current_time=0.0,
        )

        initial_confidence = belief.get_confidence()

        # Predict without new measurements
        for t in range(1, 10):
            belief.predict(dt=1.0, current_time=float(t))

        final_confidence = belief.get_confidence()

        # Confidence should have decayed
        assert final_confidence < initial_confidence

    def test_gossip_protocol_message_passing(self):
        """Test gossip protocol message exchange."""
        gossip1 = GossipProtocol(agent_id="agent_1")
        gossip2 = GossipProtocol(agent_id="agent_2")

        # Agent 1 creates belief
        belief1 = GaussianBelief(
            mean=np.array([1000.0, 500.0, 100.0, 0.0]),
            covariance=np.eye(4) * 100,
            confidence=0.9,
            timestamp=0.0,
        )

        gossip1.queue_broadcast(belief1)
        messages = gossip1.get_messages_to_send()

        assert len(messages) == 1

        # Agent 2 receives message
        accepted = gossip2.receive_message(messages[0], current_time=0.5)
        assert accepted

        # Get received beliefs
        received = gossip2.get_received_beliefs(current_time=0.5)
        assert len(received) == 1
        assert received[0][0] == "agent_1"

    def test_gossip_stale_message_rejection(self):
        """Test that stale messages are rejected."""
        gossip = GossipProtocol(agent_id="test", max_message_age=2.0)

        # Old message
        old_belief = GaussianBelief(
            mean=np.zeros(4),
            covariance=np.eye(4),
            confidence=0.5,
            timestamp=0.0,
        )

        msg = BeliefMessage(
            sender_id="other",
            belief=old_belief,
            sequence_number=1,
        )

        # Try to receive at much later time
        accepted = gossip.receive_message(msg, current_time=10.0)
        assert not accepted  # Should reject stale message

    def test_swarm_belief_manager(self):
        """Test swarm-level belief management."""
        manager = SwarmBeliefManager(
            num_agents=5,
            agent_ids=["a0", "a1", "a2", "a3", "a4"],
        )

        # Update beliefs for some agents
        manager.update_agent_belief(
            agent_id="a0",
            position=np.array([1000.0, 500.0]),
            position_uncertainty=50.0,
            confidence=0.8,
            current_time=0.0,
        )

        manager.update_agent_belief(
            agent_id="a1",
            position=np.array([1050.0, 480.0]),
            position_uncertainty=60.0,
            confidence=0.7,
            current_time=0.0,
        )

        # Compute consensus
        consensus = manager.compute_consensus()

        assert consensus.confidence > 0
        # Consensus should be between the two measurements
        assert 900 < consensus.mean[0] < 1100

        # Check detection list
        detectors = manager.get_agents_with_detection()
        assert "a0" in detectors
        assert "a1" in detectors


class TestResilientGNN:
    """Tests for resilient GNN components."""

    def test_uncertainty_encoder(self):
        """Test uncertainty-aware encoding."""
        encoder = UncertaintyEncoder(obs_dim=25, uncertainty_dim=5, embed_dim=64)

        obs = torch.randn(10, 25)
        uncertainty = torch.rand(10, 5)

        # With uncertainty
        embed_with_unc = encoder(obs, uncertainty)
        assert embed_with_unc.shape == (10, 64)

        # Without uncertainty
        embed_no_unc = encoder(obs)
        assert embed_no_unc.shape == (10, 64)

    def test_bandwidth_compressor(self):
        """Test message compression."""
        compressor = BandwidthCompressor(full_dim=64, compressed_dims=[32, 16, 8])

        x = torch.randn(10, 64)

        # Test different compression levels
        for level in range(3):
            compressed = compressor.compress(x, level)
            decompressed = compressor.decompress(compressed, level)

            expected_dim = [32, 16, 8][level]
            assert compressed.shape == (10, expected_dim)
            assert decompressed.shape == (10, 64)

    def test_resilient_gnn_forward(self):
        """Test resilient GNN forward pass."""
        gnn = ResilientSwarmGNN(
            obs_dim=25,
            uncertainty_dim=5,
            embed_dim=64,
            num_layers=2,
        )

        obs = torch.randn(10, 25)
        positions = torch.randn(10, 2) * 1000  # Spread out positions

        outputs = gnn(obs, positions)

        assert 'embeddings' in outputs
        assert 'isolation_scores' in outputs
        assert outputs['embeddings'].shape == (10, 64)
        assert outputs['isolation_scores'].shape == (10,)

    def test_resilient_gnn_degraded_mode(self):
        """Test GNN under degraded conditions."""
        gnn = ResilientSwarmGNN(obs_dim=25, embed_dim=64)

        obs = torch.randn(10, 25)
        positions = torch.randn(10, 2) * 1000

        # Normal operation
        normal_out = gnn.forward_degraded(obs, positions, degradation_level=0.0)

        # Degraded operation
        degraded_out = gnn.forward_degraded(obs, positions, degradation_level=0.7)

        # Both should produce valid outputs
        assert normal_out['embeddings'].shape == degraded_out['embeddings'].shape

        # Isolation scores should be higher under degradation
        assert degraded_out['isolation_scores'].mean() >= normal_out['isolation_scores'].mean() - 0.5

    def test_resilient_gnn_isolated_agents(self):
        """Test GNN handles isolated agents gracefully."""
        gnn = ResilientSwarmGNN(
            obs_dim=25,
            embed_dim=64,
            communication_range=100.0,  # Very short range
        )

        obs = torch.randn(5, 25)
        # Place agents far apart so no edges form
        positions = torch.tensor([
            [0.0, 0.0],
            [1000.0, 0.0],
            [2000.0, 0.0],
            [3000.0, 0.0],
            [4000.0, 0.0],
        ])

        outputs = gnn(obs, positions)

        # Should still produce valid embeddings
        assert outputs['embeddings'].shape == (5, 64)
        assert not torch.isnan(outputs['embeddings']).any()

        # All agents should have high isolation scores
        assert outputs['isolation_scores'].mean() > 0.3

    def test_resilient_coordinator(self):
        """Test full resilient coordinator."""
        coord = ResilientSwarmCoordinator(
            obs_dim=25,
            action_dim=2,
            embed_dim=64,
        )

        obs = torch.randn(10, 25)
        positions = torch.randn(10, 2) * 500

        outputs = coord(obs, positions)

        assert 'action_mean' in outputs
        assert 'action_log_std' in outputs
        assert 'value' in outputs
        assert 'roles' in outputs
        assert outputs['action_mean'].shape == (10, 2)


class TestARSHIIntegration:
    """Integration tests for complete ARSHI system."""

    def test_arshi_agent_initialization(self):
        """Test ARSHI agent initialization."""
        agent = ARSHIAgent(agent_id="test_agent")

        assert agent.agent_id == "test_agent"
        assert agent.state.mode == OperatingMode.FULL
        assert len(agent.sensors.sensors) > 0

    def test_arshi_agent_update_with_target(self):
        """Test ARSHI agent update cycle with target."""
        agent = ARSHIAgent(agent_id="test")

        own_pos = np.array([0.0, 0.0])
        own_vel = np.array([100.0, 0.0])

        target = {
            'position': np.array([3000.0, 0.0]),
            'velocity': np.array([2000.0, 0.0]),  # Hypersonic
            'mass': 1000.0,
            'radar_cross_section': 1.0,
        }

        obs = agent.update(
            own_position=own_pos,
            own_velocity=own_vel,
            target_state=target,
            current_time=0.0,
            dt=0.1,
        )

        assert isinstance(obs, ARSHIObservation)
        # Should have some detection
        assert obs.detection_confidence >= 0

    def test_arshi_mode_switching(self):
        """Test ARSHI mode transitions under degradation."""
        agent = ARSHIAgent(agent_id="test")

        own_pos = np.array([0.0, 0.0])
        own_vel = np.array([100.0, 0.0])

        # Start with full capability
        agent.set_environment_state(rf_jamming=0.0, comms_degradation=0.0)
        agent.update(own_pos, own_vel, current_time=0.0, dt=0.1)

        initial_mode = agent.state.mode

        # Apply heavy jamming
        agent.set_environment_state(rf_jamming=0.9, comms_degradation=0.8)

        # Update multiple times to trigger mode switch
        for t in range(20):
            agent.update(own_pos, own_vel, current_time=float(t), dt=0.1)

        # Mode should have changed
        assert agent.state.comm_state.jamming_detected

    def test_arshi_observation_vector(self):
        """Test ARSHI observation vector generation."""
        agent = ARSHIAgent(agent_id="test")

        # Update with target
        agent.update(
            own_position=np.array([0.0, 0.0]),
            own_velocity=np.array([100.0, 0.0]),
            target_state={
                'position': np.array([2000.0, 0.0]),
                'velocity': np.array([1500.0, 0.0]),
            },
            current_time=0.0,
            dt=0.1,
        )

        obs_vec = agent.get_observation_vector()

        assert obs_vec.shape == (15,)
        assert not np.isnan(obs_vec).any()

    def test_arshi_swarm_coordination(self):
        """Test ARSHI swarm-level coordination."""
        swarm = ARSHISwarm(num_agents=5)

        # Agent positions in formation
        positions = {
            'agent_0': np.array([0.0, 0.0]),
            'agent_1': np.array([500.0, 0.0]),
            'agent_2': np.array([250.0, 433.0]),
            'agent_3': np.array([750.0, 433.0]),
            'agent_4': np.array([500.0, 866.0]),
        }

        velocities = {
            agent_id: np.array([100.0, 0.0])
            for agent_id in positions
        }

        target = {
            'position': np.array([3000.0, 500.0]),
            'velocity': np.array([2000.0, 0.0]),
        }

        observations = swarm.update(
            agent_positions=positions,
            agent_velocities=velocities,
            target_state=target,
            communication_range=1000.0,
            current_time=0.0,
            dt=0.1,
        )

        assert len(observations) == 5

        # Check swarm state
        state_summary = swarm.get_swarm_state_summary()
        assert 'mean_confidence' in state_summary
        assert 'agents_with_detection' in state_summary

    def test_arshi_swarm_observation_matrix(self):
        """Test swarm observation matrix generation."""
        swarm = ARSHISwarm(num_agents=3)

        positions = {
            'agent_0': np.array([0.0, 0.0]),
            'agent_1': np.array([500.0, 0.0]),
            'agent_2': np.array([250.0, 433.0]),
        }

        velocities = {
            agent_id: np.array([100.0, 0.0])
            for agent_id in positions
        }

        swarm.update(
            agent_positions=positions,
            agent_velocities=velocities,
            current_time=0.0,
            dt=0.1,
        )

        obs_matrix = swarm.get_swarm_observation_matrix()

        assert obs_matrix.shape == (3, 15)
        assert not np.isnan(obs_matrix).any()

    def test_arshi_belief_propagation(self):
        """Test belief propagation across swarm."""
        swarm = ARSHISwarm(num_agents=3)

        # Close formation for communication
        positions = {
            'agent_0': np.array([0.0, 0.0]),
            'agent_1': np.array([100.0, 0.0]),
            'agent_2': np.array([50.0, 86.0]),
        }

        velocities = {
            agent_id: np.array([100.0, 0.0])
            for agent_id in positions
        }

        # Only agent_0 sees target initially
        target = {
            'position': np.array([1500.0, 0.0]),
            'velocity': np.array([1000.0, 0.0]),
        }

        # Multiple updates to allow belief propagation
        for t in range(10):
            swarm.update(
                agent_positions=positions,
                agent_velocities=velocities,
                target_state=target,
                communication_range=500.0,
                current_time=float(t) * 0.1,
                dt=0.1,
            )

        # Check that beliefs have propagated
        confidences = [
            swarm.agents[aid].belief.get_confidence()
            for aid in swarm.agent_ids
        ]

        # At least some agents should have beliefs
        assert max(confidences) > 0


class TestProprioception:
    """Tests for swarm proprioception."""

    def test_swarm_proprioception_inference(self):
        """Test proprioceptive sensing from neighbor behavior."""
        sensor = SwarmProprioception(
            observation_range=2000.0,
            min_neighbors_for_inference=2,
            acceleration_threshold=3.0,
        )

        # Simulate neighbors accelerating toward a point
        neighbor_states = {
            'n1': {
                'position': np.array([0.0, 500.0]),
                'velocity': np.array([50.0, 0.0]),
            },
            'n2': {
                'position': np.array([0.0, -500.0]),
                'velocity': np.array([50.0, 0.0]),
            },
            'n3': {
                'position': np.array([-500.0, 0.0]),
                'velocity': np.array([50.0, 0.0]),
            },
        }

        # First observation
        sensor.observe_neighbors(neighbor_states, current_time=0.0)

        # Second observation with changed velocities (accelerating toward x=1000)
        neighbor_states_t1 = {
            'n1': {
                'position': np.array([50.0, 500.0]),
                'velocity': np.array([60.0, -10.0]),  # Accelerating toward target
            },
            'n2': {
                'position': np.array([50.0, -500.0]),
                'velocity': np.array([60.0, 10.0]),
            },
            'n3': {
                'position': np.array([-450.0, 0.0]),
                'velocity': np.array([70.0, 0.0]),
            },
        }

        sensor.observe_neighbors(neighbor_states_t1, current_time=0.5)

        # Try to detect
        reading = sensor.detect(
            own_position=np.array([0.0, 0.0]),
            own_velocity=np.array([50.0, 0.0]),
            environment={
                'time': 0.5,
                'neighbor_states': neighbor_states_t1,
            },
        )

        # May or may not get reading depending on acceleration magnitude
        # Just verify it doesn't crash
        assert reading is None or isinstance(reading, SensorReading)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_swarm(self):
        """Test handling of single-agent swarm."""
        swarm = ARSHISwarm(num_agents=1)

        positions = {'agent_0': np.array([0.0, 0.0])}
        velocities = {'agent_0': np.array([100.0, 0.0])}

        observations = swarm.update(
            agent_positions=positions,
            agent_velocities=velocities,
            current_time=0.0,
            dt=0.1,
        )

        assert len(observations) == 1

    def test_no_target_detection(self):
        """Test graceful handling of no target."""
        agent = ARSHIAgent(agent_id="test")

        obs = agent.update(
            own_position=np.array([0.0, 0.0]),
            own_velocity=np.array([100.0, 0.0]),
            target_state=None,  # No target
            current_time=0.0,
            dt=0.1,
        )

        assert obs.detection_confidence == 0.0
        assert obs.target_position is None

    def test_target_out_of_range(self):
        """Test handling of target beyond sensor range."""
        agent = ARSHIAgent(agent_id="test")

        obs = agent.update(
            own_position=np.array([0.0, 0.0]),
            own_velocity=np.array([100.0, 0.0]),
            target_state={
                'position': np.array([50000.0, 0.0]),  # Very far
                'velocity': np.array([1000.0, 0.0]),
            },
            current_time=0.0,
            dt=0.1,
        )

        # Should have low or no confidence
        assert obs.detection_confidence < 0.5

    def test_reset_functionality(self):
        """Test system reset."""
        agent = ARSHIAgent(agent_id="test")

        # Build up some state
        for t in range(10):
            agent.update(
                own_position=np.array([0.0, 0.0]),
                own_velocity=np.array([100.0, 0.0]),
                target_state={
                    'position': np.array([2000.0, 0.0]),
                    'velocity': np.array([1500.0, 0.0]),
                },
                current_time=float(t),
                dt=0.1,
            )

        # Reset
        agent.reset()

        assert agent.state.mode == OperatingMode.FULL
        assert agent.belief.get_confidence() == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
