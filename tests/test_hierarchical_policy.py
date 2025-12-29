"""Tests for hierarchical role-based policies."""

import pytest
import torch
import numpy as np

from src.models.hierarchical_policy import (
    RoleType,
    HierarchicalPolicyConfig,
    RolePolicy,
    RoleAssigner,
    RoleBasedOptionCritic,
    HierarchicalMAPPO,
)


class TestRoleType:
    """Tests for RoleType enum."""

    def test_role_types_exist(self):
        """Test all role types exist."""
        assert RoleType.SCOUT == 0
        assert RoleType.TRACKER == 1
        assert RoleType.INTERCEPTOR == 2
        assert RoleType.SUPPORT == 3

    def test_role_type_count(self):
        """Test number of roles."""
        assert len(RoleType) == 4


class TestHierarchicalPolicyConfig:
    """Tests for HierarchicalPolicyConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HierarchicalPolicyConfig()
        assert config.embed_dim == 64
        assert config.hidden_dim == 128
        assert config.num_roles == 4
        assert config.role_update_interval == 10
        assert config.action_dim == 3

    def test_custom_config(self):
        """Test custom configuration."""
        config = HierarchicalPolicyConfig(
            embed_dim=128,
            hidden_dim=256,
            role_update_interval=5,
        )
        assert config.embed_dim == 128
        assert config.hidden_dim == 256
        assert config.role_update_interval == 5


class TestRolePolicy:
    """Tests for RolePolicy module."""

    @pytest.fixture
    def scout_policy(self):
        """Create a scout policy."""
        return RolePolicy(
            role=RoleType.SCOUT,
            embed_dim=64,
            hidden_dim=128,
            action_dim=3,
        )

    @pytest.fixture
    def interceptor_policy(self):
        """Create an interceptor policy."""
        return RolePolicy(
            role=RoleType.INTERCEPTOR,
            embed_dim=64,
            hidden_dim=128,
            action_dim=3,
        )

    def test_policy_creation(self, scout_policy):
        """Test policy creation."""
        assert scout_policy.role == RoleType.SCOUT
        assert scout_policy.action_dim == 3

    def test_forward_shape(self, scout_policy):
        """Test forward pass output shape."""
        batch_size = 5
        embedding = torch.randn(batch_size, 64)

        action_mean, action_log_std = scout_policy(embedding)

        assert action_mean.shape == (batch_size, 3)
        assert action_log_std.shape == (batch_size, 3)

    def test_log_std_clamped(self, scout_policy):
        """Test log_std is properly clamped."""
        embedding = torch.randn(10, 64)
        _, action_log_std = scout_policy(embedding)

        assert action_log_std.min() >= -5
        assert action_log_std.max() <= 2

    def test_role_specific_initialization(self, scout_policy, interceptor_policy):
        """Test role-specific bias initialization."""
        # Scout should have lower thrust bias than interceptor
        scout_thrust_bias = scout_policy.mean_head.bias[0].item()
        interceptor_thrust_bias = interceptor_policy.mean_head.bias[0].item()

        assert scout_thrust_bias < interceptor_thrust_bias

    def test_different_roles_different_output(self):
        """Test different roles produce different outputs."""
        torch.manual_seed(42)
        embedding = torch.randn(1, 64)

        scout = RolePolicy(RoleType.SCOUT, 64, 128, 3)
        interceptor = RolePolicy(RoleType.INTERCEPTOR, 64, 128, 3)

        scout_mean, _ = scout(embedding)
        interceptor_mean, _ = interceptor(embedding)

        # Outputs should differ due to role-specific biases
        assert not torch.allclose(scout_mean, interceptor_mean)


class TestRoleAssigner:
    """Tests for RoleAssigner module."""

    @pytest.fixture
    def assigner(self):
        """Create a role assigner."""
        return RoleAssigner(embed_dim=64, hidden_dim=128, num_roles=4)

    def test_assigner_creation(self, assigner):
        """Test assigner creation."""
        assert assigner.num_roles == 4

    def test_forward_output_shape(self, assigner):
        """Test forward pass output shapes."""
        num_agents = 10
        embeddings = torch.randn(num_agents, 64)

        role_probs, role_assignments, _ = assigner(embeddings)

        assert role_probs.shape == (num_agents, 4)
        assert role_assignments.shape == (num_agents,)

    def test_role_probs_sum_to_one(self, assigner):
        """Test role probabilities sum to 1."""
        embeddings = torch.randn(5, 64)
        role_probs, _, _ = assigner(embeddings)

        assert torch.allclose(role_probs.sum(dim=-1), torch.ones(5), atol=1e-5)

    def test_role_assignments_valid(self, assigner):
        """Test role assignments are valid indices."""
        embeddings = torch.randn(10, 64)
        _, role_assignments, _ = assigner(embeddings)

        assert role_assignments.min() >= 0
        assert role_assignments.max() < 4

    def test_return_distribution(self, assigner):
        """Test target distribution is returned when requested."""
        embeddings = torch.randn(5, 64)
        role_probs, _, target_dist = assigner(embeddings, return_distribution=True)

        assert target_dist is not None
        assert target_dist.shape == (1, 4)
        assert torch.allclose(target_dist.sum(), torch.tensor(1.0), atol=1e-5)

    def test_training_mode_uses_gumbel(self, assigner):
        """Test training mode uses Gumbel-softmax."""
        assigner.train()
        embeddings = torch.randn(5, 64)

        # Multiple forward passes in training should have some variation
        role_probs1, roles1, _ = assigner(embeddings)
        role_probs2, roles2, _ = assigner(embeddings)

        # Probs should be same (deterministic), but assignments may vary
        assert torch.allclose(role_probs1, role_probs2, atol=1e-5)

    def test_eval_mode_deterministic(self, assigner):
        """Test eval mode is deterministic."""
        assigner.eval()
        embeddings = torch.randn(5, 64)

        _, roles1, _ = assigner(embeddings)
        _, roles2, _ = assigner(embeddings)

        assert torch.equal(roles1, roles2)

    def test_role_balance_loss(self, assigner):
        """Test role balance loss computation."""
        # Create imbalanced role probs
        role_probs = torch.zeros(10, 4)
        role_probs[:, 0] = 1.0  # All agents assigned to role 0

        loss = assigner.compute_role_balance_loss(role_probs)

        # Should be high because distribution is not uniform
        assert loss.item() > 0

    def test_role_balance_loss_uniform(self, assigner):
        """Test role balance loss is low for uniform distribution."""
        # Create balanced role probs
        role_probs = torch.ones(10, 4) / 4.0

        loss = assigner.compute_role_balance_loss(role_probs)

        # Should be close to 0 for uniform distribution
        assert loss.item() < 0.1


class TestRoleBasedOptionCritic:
    """Tests for RoleBasedOptionCritic module."""

    @pytest.fixture
    def option_critic(self):
        """Create a role-based option critic."""
        config = HierarchicalPolicyConfig(
            embed_dim=64,
            hidden_dim=128,
            num_roles=4,
            action_dim=3,
        )
        return RoleBasedOptionCritic(obs_dim=79, config=config)

    def test_creation(self, option_critic):
        """Test option critic creation."""
        assert option_critic.obs_dim == 79
        assert len(option_critic.role_policies) == 4

    def test_encode(self, option_critic):
        """Test observation encoding."""
        observations = torch.randn(5, 79)
        embeddings = option_critic.encode(observations)

        assert embeddings.shape == (5, 64)

    def test_assign_roles(self, option_critic):
        """Test role assignment."""
        embeddings = torch.randn(5, 64)
        role_probs, role_assignments = option_critic.assign_roles(embeddings)

        assert role_probs.shape == (5, 4)
        assert role_assignments.shape == (5,)

    def test_role_caching(self, option_critic):
        """Test role caching behavior."""
        embeddings = torch.randn(5, 64)

        # First call should cache roles
        _, roles1 = option_critic.assign_roles(embeddings, force_update=True)
        option_critic.step_count = 1  # Simulate a step

        # Second call should use cached roles
        _, roles2 = option_critic.assign_roles(embeddings)

        assert torch.equal(roles1, roles2)

    def test_get_actions(self, option_critic):
        """Test action generation."""
        option_critic.eval()
        embeddings = torch.randn(8, 64)
        role_assignments = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])

        actions, log_probs, entropies = option_critic.get_actions(
            embeddings, role_assignments
        )

        assert actions.shape == (8, 3)
        assert log_probs.shape == (8,)
        assert entropies.shape == (8,)

    def test_action_bounds(self, option_critic):
        """Test action bounds are respected."""
        embeddings = torch.randn(10, 64)
        role_assignments = torch.randint(0, 4, (10,))

        actions, _, _ = option_critic.get_actions(embeddings, role_assignments)

        # Thrust and fire should be [0, 1]
        assert actions[:, 0].min() >= 0
        assert actions[:, 0].max() <= 1
        assert actions[:, 2].min() >= 0
        assert actions[:, 2].max() <= 1

        # Heading should be [-1, 1]
        assert actions[:, 1].min() >= -1
        assert actions[:, 1].max() <= 1

    def test_get_values(self, option_critic):
        """Test value estimation."""
        embeddings = torch.randn(5, 64)
        role_probs = torch.softmax(torch.randn(5, 4), dim=-1)

        values = option_critic.get_values(embeddings, role_probs)

        assert values.shape == (5,)

    def test_forward(self, option_critic):
        """Test full forward pass."""
        observations = torch.randn(8, 79)
        outputs = option_critic(observations)

        assert "actions" in outputs
        assert "log_probs" in outputs
        assert "values" in outputs
        assert "role_probs" in outputs
        assert "role_assignments" in outputs
        assert "entropies" in outputs

        assert outputs["actions"].shape == (8, 3)
        assert outputs["role_assignments"].shape == (8,)

    def test_forward_deterministic(self, option_critic):
        """Test deterministic forward pass."""
        option_critic.eval()
        observations = torch.randn(5, 79)

        outputs1 = option_critic(observations, deterministic=True, force_role_update=True)
        option_critic.reset_roles()
        outputs2 = option_critic(observations, deterministic=True, force_role_update=True)

        assert torch.allclose(outputs1["actions"], outputs2["actions"])

    def test_evaluate_actions(self, option_critic):
        """Test action evaluation for PPO."""
        observations = torch.randn(10, 79)
        actions = torch.rand(10, 3)
        actions[:, 1] = actions[:, 1] * 2 - 1  # Heading in [-1, 1]

        eval_outputs = option_critic.evaluate_actions(observations, actions)

        assert "values" in eval_outputs
        assert "log_probs" in eval_outputs
        assert "entropies" in eval_outputs
        assert "role_probs" in eval_outputs

    def test_reset_roles(self, option_critic):
        """Test role cache reset."""
        observations = torch.randn(5, 79)
        # Use forward which increments step_count
        option_critic(observations, force_role_update=True)

        assert option_critic.cached_roles is not None
        assert option_critic.step_count > 0

        option_critic.reset_roles()

        assert option_critic.cached_roles is None
        assert option_critic.step_count == 0


class TestHierarchicalMAPPO:
    """Tests for HierarchicalMAPPO."""

    @pytest.fixture
    def mappo(self):
        """Create HierarchicalMAPPO instance."""
        return HierarchicalMAPPO(
            obs_dim=79,
            action_dim=3,
            num_agents=5,
            device="cpu",
        )

    def test_creation(self, mappo):
        """Test MAPPO creation."""
        assert mappo.num_agents == 5
        assert mappo.policy is not None

    def test_select_actions(self, mappo):
        """Test action selection."""
        observations = {
            f"agent_{i}": np.random.randn(79).astype(np.float32)
            for i in range(5)
        }

        actions, log_probs, values, roles = mappo.select_actions(observations)

        assert len(actions) == 5
        assert len(log_probs) == 5
        assert len(values) == 5
        assert len(roles) == 5

        for aid in observations:
            assert actions[aid].shape == (3,)
            assert 0 <= roles[aid] < 4

    def test_store_transition(self, mappo):
        """Test transition storage."""
        observations = {f"agent_{i}": np.random.randn(79).astype(np.float32) for i in range(5)}
        actions = {f"agent_{i}": np.random.randn(3).astype(np.float32) for i in range(5)}
        rewards = {f"agent_{i}": np.random.randn() for i in range(5)}
        dones = {f"agent_{i}": False for i in range(5)}
        log_probs = {f"agent_{i}": np.random.randn() for i in range(5)}
        values = {f"agent_{i}": np.random.randn() for i in range(5)}
        roles = {f"agent_{i}": np.random.randint(0, 4) for i in range(5)}

        mappo.store_transition(
            observations, actions, rewards, dones, log_probs, values, roles
        )

        assert len(mappo.buffer) == 1

    def test_update_empty_buffer(self, mappo):
        """Test update with empty buffer."""
        last_values = {f"agent_{i}": 0.0 for i in range(5)}
        result = mappo.update(last_values)

        assert result == {}

    def test_update_with_data(self, mappo):
        """Test update with collected data."""
        # Collect some transitions
        for _ in range(10):
            observations = {f"agent_{i}": np.random.randn(79).astype(np.float32) for i in range(5)}
            actions = {f"agent_{i}": np.random.rand(3).astype(np.float32) for i in range(5)}
            # Adjust heading to be in [-1, 1]
            for aid in actions:
                actions[aid][1] = actions[aid][1] * 2 - 1
            rewards = {f"agent_{i}": np.random.randn() for i in range(5)}
            dones = {f"agent_{i}": False for i in range(5)}
            log_probs = {f"agent_{i}": np.random.randn() for i in range(5)}
            values = {f"agent_{i}": np.random.randn() for i in range(5)}
            roles = {f"agent_{i}": np.random.randint(0, 4) for i in range(5)}

            mappo.store_transition(
                observations, actions, rewards, dones, log_probs, values, roles
            )

        last_values = {f"agent_{i}": 0.0 for i in range(5)}
        result = mappo.update(last_values, num_epochs=2)

        assert "policy_loss" in result
        assert "value_loss" in result
        assert "entropy" in result
        assert "role_entropy" in result

        # Buffer should be cleared
        assert len(mappo.buffer) == 0

    def test_gae_computation(self, mappo):
        """Test GAE computation."""
        # Store a few transitions
        for _ in range(5):
            observations = {f"agent_{i}": np.random.randn(79).astype(np.float32) for i in range(5)}
            actions = {f"agent_{i}": np.random.randn(3).astype(np.float32) for i in range(5)}
            rewards = {f"agent_{i}": 1.0 for i in range(5)}  # Constant reward
            dones = {f"agent_{i}": False for i in range(5)}
            log_probs = {f"agent_{i}": 0.0 for i in range(5)}
            values = {f"agent_{i}": 0.5 for i in range(5)}
            roles = {f"agent_{i}": 0 for i in range(5)}

            mappo.store_transition(
                observations, actions, rewards, dones, log_probs, values, roles
            )

        last_values = {f"agent_{i}": 0.0 for i in range(5)}
        advantages, returns = mappo._compute_gae(last_values, gamma=0.99, gae_lambda=0.95)

        assert len(advantages) == 5
        assert len(returns) == 5
        assert len(advantages[0]) == 5  # num_agents

    def test_save_load(self, mappo, tmp_path):
        """Test model save and load."""
        save_path = str(tmp_path / "model.pt")

        # Set seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        # Use fixed observations
        observations = {f"agent_{i}": np.zeros(79, dtype=np.float32) + i * 0.1 for i in range(5)}

        # Reset roles and get deterministic outputs
        mappo.policy.reset_roles()
        mappo.policy.eval()
        actions1, _, _, roles1 = mappo.select_actions(observations, deterministic=True)

        # Save
        mappo.save(save_path)

        # Create new model and load
        mappo2 = HierarchicalMAPPO(
            obs_dim=79,
            action_dim=3,
            num_agents=5,
            device="cpu",
        )
        mappo2.load(save_path)

        # Get outputs from loaded model with same state
        mappo2.policy.reset_roles()
        mappo2.policy.eval()
        actions2, _, _, roles2 = mappo2.select_actions(observations, deterministic=True)

        # Roles should be the same (deterministic in eval mode)
        for aid in observations:
            assert roles1[aid] == roles2[aid], f"Roles differ for {aid}: {roles1[aid]} vs {roles2[aid]}"
            assert np.allclose(actions1[aid], actions2[aid], atol=1e-5), f"Actions differ for {aid}"


class TestIntegration:
    """Integration tests for hierarchical policies."""

    def test_full_episode_simulation(self):
        """Test a simulated episode with role-based policies."""
        mappo = HierarchicalMAPPO(
            obs_dim=79,
            action_dim=3,
            num_agents=10,
            device="cpu",
        )

        # Simulate an episode
        for step in range(20):
            observations = {
                f"agent_{i}": np.random.randn(79).astype(np.float32)
                for i in range(10)
            }

            actions, log_probs, values, roles = mappo.select_actions(observations)

            # Verify role diversity
            unique_roles = set(roles.values())
            # With 10 agents, we should see multiple roles
            # (though not guaranteed due to random assignment)

            # Create fake rewards and dones
            rewards = {f"agent_{i}": 1.0 for i in range(10)}
            dones = {f"agent_{i}": step == 19 for i in range(10)}

            mappo.store_transition(
                observations, actions, rewards, dones, log_probs, values, roles
            )

        # Update policy
        last_values = {f"agent_{i}": 0.0 for i in range(10)}
        stats = mappo.update(last_values, num_epochs=2)

        assert stats["policy_loss"] is not None
        assert stats["value_loss"] is not None

    def test_role_stability_over_episode(self):
        """Test that roles remain stable within update interval."""
        config = HierarchicalPolicyConfig(role_update_interval=5)
        option_critic = RoleBasedOptionCritic(obs_dim=79, config=config)
        option_critic.eval()

        observations = torch.randn(5, 79)

        # First forward - assigns roles
        outputs1 = option_critic(observations, force_role_update=True)
        initial_roles = outputs1["role_assignments"].clone()

        # Next few forwards should use cached roles
        for _ in range(4):
            outputs = option_critic(observations)
            assert torch.equal(outputs["role_assignments"], initial_roles)

        # After update interval, roles may change
        option_critic.step_count = 5
        outputs_after = option_critic(observations)
        # Roles should still be valid
        assert outputs_after["role_assignments"].min() >= 0
        assert outputs_after["role_assignments"].max() < 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
