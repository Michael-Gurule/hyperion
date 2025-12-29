"""
Comprehensive tests for scaled HYPERION components.

Tests:
- GNN communication layer
- MAPPO algorithm
- Scaled environment
- Adaptive sensor fusion
"""

import pytest
import numpy as np
import torch
from typing import Dict, List

# Import components to test
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.gnn_communication import (
    AgentNodeEncoder,
    GNNCommunicationLayer,
    SwarmGNN,
    CoordinationHead,
    SwarmCoordinationNetwork,
)
from src.models.adaptive_sensor_fusion import (
    AdaptiveKalmanFilter,
    BayesianSensorFusion,
    MultiHypothesisTracker,
    SensorMeasurement,
    SensorType,
)
from src.env.scaled_environment import (
    ScaledHypersonicSwarmEnv,
    create_scaled_env,
    AdversarialConfig,
    RewardConfig,
    TargetBehavior,
    AgentRole,
)
from src.training.mappo import MAPPO, MAPPOConfig, ActorNetwork, CentralizedCritic


# ============================================================================
# GNN Communication Tests
# ============================================================================


class TestAgentNodeEncoder:
    """Tests for AgentNodeEncoder."""

    def test_encoder_creation(self):
        """Test encoder initialization."""
        encoder = AgentNodeEncoder(obs_dim=25, hidden_dim=128, embed_dim=64)
        assert encoder.embed_dim == 64

    def test_forward_pass(self):
        """Test forward pass with various input shapes."""
        encoder = AgentNodeEncoder(obs_dim=25, hidden_dim=128, embed_dim=64)

        # Single observation
        obs = torch.randn(25)
        out = encoder(obs, use_full_obs=True)
        assert out.shape == (1, 64)

        # Batch of observations
        obs_batch = torch.randn(10, 25)
        out_batch = encoder(obs_batch, use_full_obs=True)
        assert out_batch.shape == (10, 64)

    def test_self_encoder(self):
        """Test self-state encoder branch."""
        encoder = AgentNodeEncoder(obs_dim=25, hidden_dim=128, embed_dim=64)
        obs = torch.randn(5, 25)
        out = encoder(obs, use_full_obs=False)
        assert out.shape == (5, 64)


class TestGNNCommunicationLayer:
    """Tests for GNN communication layer."""

    def test_layer_creation(self):
        """Test layer initialization."""
        layer = GNNCommunicationLayer(
            node_feat_dim=64, edge_dim=8, hidden_dim=128, heads=4
        )
        assert layer.node_feat_dim == 64
        assert layer.heads == 4

    def test_message_passing(self):
        """Test message passing between nodes."""
        layer = GNNCommunicationLayer(node_feat_dim=64, edge_dim=8, hidden_dim=128, heads=4)

        num_nodes = 10
        x = torch.randn(num_nodes, 64)

        # Create edge index (fully connected)
        edge_index = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index.append([i, j])
        edge_index = torch.tensor(edge_index).t()

        edge_attr = torch.randn(edge_index.size(1), 8)

        out = layer(x, edge_index, edge_attr)
        assert out.shape == x.shape

    def test_empty_graph(self):
        """Test with no edges."""
        layer = GNNCommunicationLayer(node_feat_dim=64, edge_dim=8, hidden_dim=128, heads=4)

        x = torch.randn(5, 64)
        edge_index = torch.tensor([[], []], dtype=torch.long)

        out = layer(x, edge_index)
        assert out.shape == x.shape


class TestSwarmGNN:
    """Tests for full SwarmGNN."""

    def test_gnn_creation(self):
        """Test GNN initialization."""
        gnn = SwarmGNN(
            obs_dim=25,
            hidden_dim=128,
            embed_dim=64,
            num_gnn_layers=3,
            communication_range=1500.0,
        )
        assert len(gnn.gnn_layers) == 3

    def test_graph_building(self):
        """Test graph construction from positions."""
        gnn = SwarmGNN(obs_dim=25, communication_range=1000.0)

        # Create positions in a line
        positions = torch.tensor([
            [0.0, 0.0],
            [500.0, 0.0],  # Within range
            [1500.0, 0.0],  # Out of range
        ])

        edge_index, edge_attr = gnn.build_graph(positions)

        # Should have edges between nodes 0 and 1 only
        assert edge_index.size(1) == 2  # Bidirectional

    def test_forward_pass(self):
        """Test full forward pass."""
        gnn = SwarmGNN(
            obs_dim=75,  # Scaled env observation size
            hidden_dim=128,
            embed_dim=64,
            num_gnn_layers=2,
            communication_range=2000.0,
        )

        num_agents = 20
        obs = torch.randn(num_agents, 75)
        positions = torch.randn(num_agents, 2) * 5000

        out = gnn(obs, positions)
        assert out.shape == (num_agents, 64)


class TestSwarmCoordinationNetwork:
    """Tests for complete coordination network."""

    def test_network_creation(self):
        """Test network initialization."""
        net = SwarmCoordinationNetwork(
            obs_dim=75,
            action_dim=2,
            hidden_dim=128,
            embed_dim=64,
            num_gnn_layers=2,
            num_roles=4,
        )
        assert net.action_dim == 2

    def test_forward_output(self):
        """Test forward pass outputs."""
        net = SwarmCoordinationNetwork(
            obs_dim=75,
            action_dim=2,
            hidden_dim=128,
            embed_dim=64,
        )

        obs = torch.randn(10, 75)
        outputs = net(obs)

        assert "action_mean" in outputs
        assert "action_log_std" in outputs
        assert "value" in outputs
        assert "roles" in outputs
        assert outputs["action_mean"].shape == (10, 2)

    def test_get_action(self):
        """Test action sampling."""
        net = SwarmCoordinationNetwork(obs_dim=75, action_dim=2)

        obs = torch.randn(10, 75)

        # Stochastic action
        actions, log_probs = net.get_action(obs, deterministic=False)
        assert actions.shape == (10, 2)
        assert log_probs.shape == (10,)

        # Deterministic action
        det_actions, _ = net.get_action(obs, deterministic=True)
        assert det_actions.shape == (10, 2)


# ============================================================================
# MAPPO Tests
# ============================================================================


class TestActorNetwork:
    """Tests for actor network."""

    def test_actor_creation(self):
        """Test actor initialization."""
        actor = ActorNetwork(obs_dim=25, action_dim=2, hidden_dim=128)
        assert actor.obs_dim == 25
        assert actor.action_dim == 2

    def test_forward_pass(self):
        """Test forward pass."""
        actor = ActorNetwork(obs_dim=25, action_dim=2)

        obs = torch.randn(5, 25)
        mean, std = actor(obs)

        assert mean.shape == (5, 2)
        assert std.shape == (2,)
        assert (std > 0).all()

    def test_get_action(self):
        """Test action sampling."""
        actor = ActorNetwork(obs_dim=25, action_dim=2)

        obs = torch.randn(5, 25)
        action, log_prob = actor.get_action(obs)

        assert action.shape == (5, 2)
        assert log_prob.shape == (5,)

        # Check action bounds
        assert (action[:, 0] >= 0).all() and (action[:, 0] <= 1).all()


class TestCentralizedCritic:
    """Tests for centralized critic."""

    def test_critic_creation(self):
        """Test critic initialization."""
        critic = CentralizedCritic(
            obs_dim=25,
            max_agents=50,
            hidden_dim=128,
        )
        assert critic.obs_dim == 25

    def test_forward_pass(self):
        """Test forward pass with global observations."""
        critic = CentralizedCritic(obs_dim=25, max_agents=50)

        all_obs = torch.randn(4, 10, 25)  # 4 batches, 10 agents
        values = critic(all_obs)

        assert values.shape == (4, 10)


class TestMAPPO:
    """Tests for MAPPO algorithm."""

    def test_mappo_creation(self):
        """Test MAPPO initialization."""
        mappo = MAPPO(
            obs_dim=25,
            action_dim=2,
            num_agents=5,
        )
        assert mappo.obs_dim == 25
        assert mappo.num_agents == 5

    def test_select_actions(self):
        """Test action selection."""
        mappo = MAPPO(obs_dim=25, action_dim=2, num_agents=5)

        observations = {f"agent_{i}": np.random.randn(25) for i in range(5)}
        actions, log_probs, values = mappo.select_actions(observations)

        assert len(actions) == 5
        assert len(log_probs) == 5
        assert len(values) == 5

        for agent_id in observations.keys():
            assert actions[agent_id].shape == (2,)

    def test_store_and_update(self):
        """Test storing transitions and updating."""
        mappo = MAPPO(obs_dim=25, action_dim=2, num_agents=3)

        # Store some transitions
        for _ in range(10):
            obs = {f"agent_{i}": np.random.randn(25) for i in range(3)}
            actions, log_probs, values = mappo.select_actions(obs)
            rewards = {f"agent_{i}": np.random.randn() for i in range(3)}
            dones = {f"agent_{i}": False for i in range(3)}

            mappo.store_transition(obs, actions, rewards, dones, log_probs, values)

        assert len(mappo.buffer) == 10

        # Update
        last_values = {f"agent_{i}": 0.0 for i in range(3)}
        stats = mappo.update(last_values)

        assert "policy_loss" in stats
        assert "value_loss" in stats
        assert len(mappo.buffer) == 0  # Buffer cleared


# ============================================================================
# Scaled Environment Tests
# ============================================================================


class TestScaledEnvironment:
    """Tests for scaled environment."""

    def test_env_creation(self):
        """Test environment initialization."""
        env = ScaledHypersonicSwarmEnv(num_agents=20)
        assert env.num_agents == 20
        assert len(env.possible_agents) == 20

    def test_observation_space(self):
        """Test observation space dimensions."""
        # Test without projectiles (backward compatibility)
        env = ScaledHypersonicSwarmEnv(
            num_agents=20, max_observed_neighbors=10, use_projectiles=False
        )
        obs_space = env.observation_space(env.possible_agents[0])
        # Expected: 5 self + 6 target + 10*6 neighbors + 4 global = 75
        expected_dim = 5 + 6 + 10 * 6 + 4
        assert obs_space.shape[0] == expected_dim

    def test_observation_space_with_projectiles(self):
        """Test observation space with projectiles enabled."""
        env = ScaledHypersonicSwarmEnv(
            num_agents=20, max_observed_neighbors=10, use_projectiles=True
        )
        obs_space = env.observation_space(env.possible_agents[0])
        # Expected: 5 self + 6 target + 10*6 neighbors + 4 global + 4 projectile = 79
        expected_dim = 5 + 6 + 10 * 6 + 4 + 4
        assert obs_space.shape[0] == expected_dim

    def test_action_space(self):
        """Test action space without projectiles."""
        env = ScaledHypersonicSwarmEnv(num_agents=10, use_projectiles=False)
        act_space = env.action_space(env.possible_agents[0])
        assert act_space.shape[0] == 2

    def test_action_space_with_projectiles(self):
        """Test action space with projectiles enabled."""
        env = ScaledHypersonicSwarmEnv(num_agents=10, use_projectiles=True)
        act_space = env.action_space(env.possible_agents[0])
        # With projectiles: [thrust, heading, fire]
        assert act_space.shape[0] == 3

    def test_reset(self):
        """Test environment reset without projectiles."""
        env = ScaledHypersonicSwarmEnv(num_agents=30, use_projectiles=False)

        observations, infos = env.reset(seed=42)

        assert len(observations) == 30
        assert all(obs.shape == (75,) for obs in observations.values())

    def test_reset_with_projectiles(self):
        """Test environment reset with projectiles enabled."""
        env = ScaledHypersonicSwarmEnv(num_agents=30, use_projectiles=True)

        observations, infos = env.reset(seed=42)

        assert len(observations) == 30
        assert all(obs.shape == (79,) for obs in observations.values())

    def test_step(self):
        """Test environment step."""
        env = ScaledHypersonicSwarmEnv(num_agents=10)

        observations, _ = env.reset()
        actions = {
            agent: env.action_space(agent).sample() for agent in env.agents
        }

        next_obs, rewards, terms, truncs, infos = env.step(actions)

        assert len(next_obs) == 10 or len(next_obs) == 0
        assert len(rewards) == 10
        assert all(isinstance(r, float) for r in rewards.values())

    def test_adversarial_target(self):
        """Test adversarial target behavior."""
        config = AdversarialConfig(
            enabled=True,
            evasive_maneuvers=True,
            evasion_probability=1.0,
        )
        env = ScaledHypersonicSwarmEnv(
            num_agents=5,
            adversarial_config=config,
        )

        env.reset()

        # Run some steps
        initial_pos = env.target_state["position"].copy()
        for _ in range(10):
            actions = {a: env.action_space(a).sample() for a in env.agents}
            env.step(actions)

        # Target should have moved
        assert not np.allclose(env.target_state["position"], initial_pos)

    def test_role_assignment(self):
        """Test agent role assignment."""
        env = ScaledHypersonicSwarmEnv(num_agents=20)
        env.reset()

        roles = [env.agent_states[a]["role"] for a in env.agents]

        # Should have multiple roles
        unique_roles = set(roles)
        assert len(unique_roles) > 1

    def test_coordination_rewards(self):
        """Test coordination-aware rewards."""
        reward_config = RewardConfig(
            formation_bonus=0.5,
            role_bonus=0.5,
        )
        env = ScaledHypersonicSwarmEnv(
            num_agents=10,
            reward_config=reward_config,
            use_projectiles=False,  # Use 2D actions for this test
        )

        env.reset()
        actions = {a: np.array([0.5, 0.0]) for a in env.agents}
        _, rewards, _, _, _ = env.step(actions)

        # Rewards should vary based on coordination
        assert len(set(rewards.values())) > 1


class TestCreateScaledEnv:
    """Tests for environment factory."""

    def test_factory_creation(self):
        """Test factory function."""
        env = create_scaled_env(num_agents=50, adversarial=True)
        assert env.num_agents == 50
        assert env.adversarial.enabled

    def test_factory_no_adversarial(self):
        """Test without adversarial."""
        env = create_scaled_env(num_agents=30, adversarial=False)
        assert not env.adversarial.enabled


# ============================================================================
# Adaptive Sensor Fusion Tests
# ============================================================================


class TestAdaptiveKalmanFilter:
    """Tests for adaptive Kalman filter."""

    def test_filter_creation(self):
        """Test filter initialization."""
        kf = AdaptiveKalmanFilter(state_dim=6, measurement_dim=2)
        assert kf.state_dim == 6
        assert not kf.initialized

    def test_predict(self):
        """Test prediction step."""
        kf = AdaptiveKalmanFilter(state_dim=6)
        kf.state = np.array([0, 0, 100, 50, 0, 0])  # Moving target
        kf.initialized = True

        kf.predict(dt=0.1)

        # Position should have updated based on velocity
        assert kf.state[0] > 0  # x increased
        assert kf.state[1] > 0  # y increased

    def test_update(self):
        """Test measurement update."""
        kf = AdaptiveKalmanFilter(state_dim=4)

        # Initialize with measurement
        kf.update(np.array([100.0, 200.0]))
        assert kf.initialized
        assert np.allclose(kf.state[:2], [100.0, 200.0])

        # Update with new measurement
        kf.predict(dt=0.1)
        accepted, mahal = kf.update(np.array([110.0, 210.0]))

        assert accepted
        assert mahal >= 0

    def test_outlier_rejection(self):
        """Test outlier detection."""
        kf = AdaptiveKalmanFilter(state_dim=4)

        kf.update(np.array([0.0, 0.0]))
        kf.predict(dt=0.1)

        # Normal update
        accepted1, _ = kf.update(np.array([10.0, 10.0]))
        assert accepted1

        kf.predict(dt=0.1)

        # Outlier (very far from expected)
        accepted2, mahal = kf.update(
            np.array([10000.0, 10000.0]),
            outlier_threshold=5.0,
        )
        assert not accepted2
        assert mahal > 5.0

    def test_uncertainty_quantification(self):
        """Test uncertainty estimates."""
        kf = AdaptiveKalmanFilter(state_dim=6)

        # Initial high uncertainty
        kf.update(np.array([0.0, 0.0]))
        initial_uncertainty = kf.get_position_uncertainty()

        # Uncertainty should decrease with more measurements
        for _ in range(10):
            kf.predict(dt=0.1)
            kf.update(np.array([0.0, 0.0]) + np.random.randn(2) * 1.0)

        final_uncertainty = kf.get_position_uncertainty()
        assert final_uncertainty < initial_uncertainty


class TestBayesianSensorFusion:
    """Tests for Bayesian sensor fusion."""

    def test_fusion_creation(self):
        """Test fusion system initialization."""
        fusion = BayesianSensorFusion(
            sensor_types=[SensorType.RF, SensorType.THERMAL]
        )
        assert len(fusion.sensor_types) == 2

    def test_reliability_estimation(self):
        """Test sensor reliability tracking."""
        fusion = BayesianSensorFusion()

        initial_reliability = fusion.get_sensor_reliability(SensorType.RF)

        # Update with accepted measurements
        for _ in range(10):
            fusion.update_sensor_reliability(SensorType.RF, True)

        improved_reliability = fusion.get_sensor_reliability(SensorType.RF)
        assert improved_reliability >= initial_reliability

    def test_measurement_fusion(self):
        """Test fusing multiple measurements."""
        fusion = BayesianSensorFusion()

        measurements = [
            SensorMeasurement(
                position=np.array([100.0, 100.0]),
                sensor_type=SensorType.RF,
                timestamp=0.0,
                raw_confidence=0.8,
            ),
            SensorMeasurement(
                position=np.array([105.0, 95.0]),
                sensor_type=SensorType.THERMAL,
                timestamp=0.0,
                raw_confidence=0.6,
            ),
        ]

        fused_pos, confidence, contributions = fusion.fuse_measurements(
            measurements, dt=0.1
        )

        assert fused_pos.shape == (2,)
        assert 0 <= confidence <= 1
        assert "rf" in contributions or "thermal" in contributions

    def test_confidence_interval(self):
        """Test confidence interval computation."""
        fusion = BayesianSensorFusion()

        meas = SensorMeasurement(
            position=np.array([0.0, 0.0]),
            sensor_type=SensorType.RF,
            timestamp=0.0,
        )
        fusion.fuse_measurements([meas], dt=0.1)

        lower, upper = fusion.get_confidence_interval(confidence_level=0.95)

        assert (lower < upper).all()
        pos = fusion.get_position()
        assert (pos >= lower).all() and (pos <= upper).all()


class TestMultiHypothesisTracker:
    """Tests for multi-hypothesis tracker."""

    def test_tracker_creation(self):
        """Test tracker initialization."""
        tracker = MultiHypothesisTracker(max_tracks=20)
        assert tracker.max_tracks == 20
        assert len(tracker.tracks) == 0

    def test_track_creation(self):
        """Test creating new tracks."""
        tracker = MultiHypothesisTracker(confirmation_threshold=2)

        measurements = [
            SensorMeasurement(
                position=np.array([100.0, 100.0]),
                sensor_type=SensorType.RF,
                timestamp=0.0,
            )
        ]

        confirmed = tracker.update(measurements, dt=0.1)
        assert len(confirmed) == 0  # Not yet confirmed
        assert len(tracker.tracks) == 1

    def test_track_confirmation(self):
        """Test track confirmation."""
        tracker = MultiHypothesisTracker(
            confirmation_threshold=3,
        )

        # Use stationary measurements so they always associate to same track
        for i in range(5):
            measurements = [
                SensorMeasurement(
                    position=np.array([100.0, 100.0]),  # Stationary target
                    sensor_type=SensorType.RF,
                    timestamp=i * 0.1,
                )
            ]
            tracker.update(measurements, dt=0.1)

        # Track should be confirmed after enough hits
        tracks = tracker.get_all_tracks()
        assert len(tracks) >= 1
        # At least one track should have enough hits to be confirmed
        assert any(t.hits >= 3 for t in tracks)

    def test_track_deletion(self):
        """Test stale track deletion."""
        tracker = MultiHypothesisTracker(
            confirmation_threshold=1,
            deletion_threshold=3,
        )

        # Create track
        measurements = [
            SensorMeasurement(
                position=np.array([100.0, 100.0]),
                sensor_type=SensorType.RF,
                timestamp=0.0,
            )
        ]
        tracker.update(measurements, dt=0.1)

        # Miss updates (no measurements)
        for _ in range(5):
            tracker.update([], dt=0.1)

        assert len(tracker.tracks) == 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for combined components."""

    def test_gnn_with_env(self):
        """Test GNN with environment observations."""
        env = create_scaled_env(num_agents=20)
        observations, _ = env.reset()

        obs_dim = list(observations.values())[0].shape[0]
        gnn = SwarmGNN(obs_dim=obs_dim)

        obs_tensor = torch.tensor(
            np.array(list(observations.values())), dtype=torch.float32
        )

        embeddings = gnn(obs_tensor)
        assert embeddings.shape[0] == 20

    def test_mappo_with_env(self):
        """Test MAPPO with scaled environment."""
        env = create_scaled_env(num_agents=10)
        observations, _ = env.reset()

        obs_dim = list(observations.values())[0].shape[0]
        mappo = MAPPO(obs_dim=obs_dim, action_dim=2, num_agents=10)

        actions, _, _ = mappo.select_actions(observations)

        next_obs, rewards, _, _, _ = env.step(actions)
        assert len(rewards) == 10

    def test_sensor_fusion_with_env(self):
        """Test sensor fusion with environment data."""
        env = create_scaled_env(num_agents=5)
        observations, _ = env.reset()

        fusion = BayesianSensorFusion()

        # Create measurements from environment
        target_pos = env.target_state["position"]
        measurements = [
            SensorMeasurement(
                position=target_pos + np.random.randn(2) * 50,
                sensor_type=SensorType.RF,
                timestamp=0.0,
                raw_confidence=0.8,
            ),
        ]

        fused_pos, conf, _ = fusion.fuse_measurements(measurements, dt=0.1)

        # Fused position should be close to true target
        error = np.linalg.norm(fused_pos - target_pos)
        assert error < 200  # Within reasonable error


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
