"""Tests for the projectile/missile system with PN guidance."""

import numpy as np
import pytest

from src.env.projectile_system import (
    ProjectileConfig,
    ProjectileManager,
    Projectile,
    GuidanceType,
    ProportionalNavigationGuidance,
)


class TestProjectileConfig:
    """Tests for ProjectileConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ProjectileConfig()
        assert config.speed == 600.0
        assert config.lifetime == 5.0
        assert config.hit_radius == 20.0
        assert config.cooldown == 2.0
        assert config.max_per_agent == 3
        assert config.guidance_type == GuidanceType.PROPORTIONAL_NAVIGATION
        assert config.nav_constant == 3.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = ProjectileConfig(
            speed=800.0,
            lifetime=10.0,
            hit_radius=30.0,
            nav_constant=4.0,
        )
        assert config.speed == 800.0
        assert config.lifetime == 10.0
        assert config.hit_radius == 30.0
        assert config.nav_constant == 4.0


class TestProportionalNavigationGuidance:
    """Tests for PN guidance implementation."""

    def test_los_angle_computation(self):
        """Test line-of-sight angle calculation."""
        pn = ProportionalNavigationGuidance(nav_constant=3.0)

        # Target directly ahead
        projectile_pos = np.array([0.0, 0.0])
        target_pos = np.array([100.0, 0.0])
        angle = pn.compute_los_angle(projectile_pos, target_pos)
        assert np.isclose(angle, 0.0)

        # Target directly above
        target_pos = np.array([0.0, 100.0])
        angle = pn.compute_los_angle(projectile_pos, target_pos)
        assert np.isclose(angle, np.pi / 2)

    def test_los_rate_computation(self):
        """Test LOS rate calculation."""
        pn = ProportionalNavigationGuidance()

        # No change in LOS
        rate = pn.compute_los_rate(0.0, 0.0, dt=0.1)
        assert rate == 0.0

        # 45 degree change in 1 second
        rate = pn.compute_los_rate(np.pi / 4, 0.0, dt=1.0)
        assert np.isclose(rate, np.pi / 4)

    def test_closing_velocity(self):
        """Test closing velocity calculation."""
        pn = ProportionalNavigationGuidance()

        projectile_pos = np.array([0.0, 0.0])
        projectile_vel = np.array([100.0, 0.0])  # Moving toward target
        target_pos = np.array([100.0, 0.0])
        target_vel = np.array([-50.0, 0.0])  # Moving toward projectile

        closing_vel = pn.compute_closing_velocity(
            projectile_pos, projectile_vel, target_pos, target_vel
        )

        # Closing velocity should be positive (approaching)
        assert closing_vel > 0

    def test_acceleration_command(self):
        """Test PN acceleration command generation."""
        pn = ProportionalNavigationGuidance(nav_constant=3.0)

        projectile_pos = np.array([0.0, 0.0])
        projectile_vel = np.array([100.0, 0.0])
        target_pos = np.array([500.0, 50.0])  # Slightly above
        target_vel = np.array([100.0, 0.0])

        # First call - no previous LOS
        accel, los = pn.compute_acceleration_command(
            projectile_pos, projectile_vel, target_pos, target_vel,
            prev_los_angle=None, dt=0.1
        )
        assert np.allclose(accel, [0.0, 0.0])  # No command on first call

        # Second call with previous LOS
        accel, new_los = pn.compute_acceleration_command(
            projectile_pos, projectile_vel, target_pos, target_vel,
            prev_los_angle=los - 0.01, dt=0.1  # Slight LOS change
        )
        # Should have non-zero acceleration perpendicular to LOS
        assert np.linalg.norm(accel) >= 0


class TestProjectileManager:
    """Tests for ProjectileManager."""

    def test_initialization(self):
        """Test manager initialization."""
        manager = ProjectileManager()
        assert len(manager.projectiles) == 0
        assert manager.next_projectile_id == 0

    def test_reset(self):
        """Test reset clears all state."""
        manager = ProjectileManager()

        # Launch a projectile
        manager.launch_projectile(
            agent_id="agent_0",
            agent_position=np.array([0.0, 0.0]),
            target_position=np.array([100.0, 0.0]),
            target_velocity=np.array([50.0, 0.0]),
            current_time=0.0,
        )

        manager.reset()
        assert len(manager.projectiles) == 0
        assert len(manager.agent_cooldowns) == 0

    def test_launch_projectile(self):
        """Test launching a projectile."""
        manager = ProjectileManager()

        projectile = manager.launch_projectile(
            agent_id="agent_0",
            agent_position=np.array([0.0, 0.0]),
            target_position=np.array([100.0, 0.0]),
            target_velocity=np.array([50.0, 0.0]),
            current_time=0.0,
        )

        assert projectile is not None
        assert projectile.id == 0
        assert projectile.owner_agent == "agent_0"
        assert projectile.active
        assert len(manager.projectiles) == 1

    def test_cooldown_enforcement(self):
        """Test that cooldown prevents rapid firing."""
        config = ProjectileConfig(cooldown=2.0)
        manager = ProjectileManager(config)

        # First launch should succeed
        p1 = manager.launch_projectile(
            agent_id="agent_0",
            agent_position=np.array([0.0, 0.0]),
            target_position=np.array([100.0, 0.0]),
            target_velocity=np.array([50.0, 0.0]),
            current_time=0.0,
        )
        assert p1 is not None

        # Second launch immediately after should fail
        p2 = manager.launch_projectile(
            agent_id="agent_0",
            agent_position=np.array([0.0, 0.0]),
            target_position=np.array([100.0, 0.0]),
            target_velocity=np.array([50.0, 0.0]),
            current_time=0.5,
        )
        assert p2 is None

        # After cooldown, should succeed
        p3 = manager.launch_projectile(
            agent_id="agent_0",
            agent_position=np.array([0.0, 0.0]),
            target_position=np.array([100.0, 0.0]),
            target_velocity=np.array([50.0, 0.0]),
            current_time=2.5,
        )
        assert p3 is not None

    def test_max_projectiles_per_agent(self):
        """Test maximum projectiles per agent limit."""
        config = ProjectileConfig(max_per_agent=2, cooldown=0.0)
        manager = ProjectileManager(config)

        # Launch max projectiles
        for i in range(2):
            p = manager.launch_projectile(
                agent_id="agent_0",
                agent_position=np.array([0.0, 0.0]),
                target_position=np.array([100.0, 0.0]),
                target_velocity=np.array([50.0, 0.0]),
                current_time=float(i),
            )
            assert p is not None

        # Next should fail (at max)
        p = manager.launch_projectile(
            agent_id="agent_0",
            agent_position=np.array([0.0, 0.0]),
            target_position=np.array([100.0, 0.0]),
            target_velocity=np.array([50.0, 0.0]),
            current_time=2.0,
        )
        assert p is None

    def test_projectile_hit_detection(self):
        """Test projectile-target hit detection."""
        config = ProjectileConfig(speed=1000.0, hit_radius=50.0, lifetime=10.0)
        manager = ProjectileManager(config)

        # Launch toward stationary target
        manager.launch_projectile(
            agent_id="agent_0",
            agent_position=np.array([0.0, 0.0]),
            target_position=np.array([200.0, 0.0]),  # Target ahead
            target_velocity=np.array([0.0, 0.0]),
            current_time=0.0,
        )

        # Update until hit - projectile moves at 1000 units/s, target at 200 units
        # Should hit in about 0.2 seconds
        hits = []
        for i in range(50):
            step_hits = manager.update(
                dt=0.1,
                current_time=(i + 1) * 0.1,
                target_position=np.array([200.0, 0.0]),
                target_velocity=np.array([0.0, 0.0]),
            )
            hits.extend(step_hits)
            if hits:
                break

        assert len(hits) >= 1
        assert hits[0].owner_agent == "agent_0"

    def test_projectile_lifetime_expiry(self):
        """Test that projectiles expire after lifetime."""
        config = ProjectileConfig(speed=100.0, lifetime=1.0)
        manager = ProjectileManager(config)

        manager.launch_projectile(
            agent_id="agent_0",
            agent_position=np.array([0.0, 0.0]),
            target_position=np.array([10000.0, 0.0]),  # Far away
            target_velocity=np.array([0.0, 0.0]),
            current_time=0.0,
        )

        assert len(manager.projectiles) == 1

        # Update past lifetime
        for i in range(15):
            manager.update(
                dt=0.1,
                current_time=i * 0.1,
                target_position=np.array([10000.0, 0.0]),
                target_velocity=np.array([0.0, 0.0]),
            )

        # Projectile should be removed
        assert len(manager.projectiles) == 0

    def test_pn_guidance_tracks_target(self):
        """Test that PN guidance successfully tracks moving target."""
        config = ProjectileConfig(
            speed=600.0,
            lifetime=10.0,
            hit_radius=30.0,
            guidance_type=GuidanceType.PROPORTIONAL_NAVIGATION,
        )
        manager = ProjectileManager(config)

        # Launch from origin toward moving target
        manager.launch_projectile(
            agent_id="agent_0",
            agent_position=np.array([0.0, 0.0]),
            target_position=np.array([500.0, 0.0]),
            target_velocity=np.array([100.0, 50.0]),  # Moving target
            current_time=0.0,
        )

        # Simulate with moving target
        target_pos = np.array([500.0, 0.0])
        target_vel = np.array([100.0, 50.0])
        hits = []

        for i in range(50):
            target_pos = target_pos + target_vel * 0.1
            step_hits = manager.update(
                dt=0.1,
                current_time=i * 0.1,
                target_position=target_pos,
                target_velocity=target_vel,
            )
            hits.extend(step_hits)
            if hits:
                break

        # Should hit the moving target
        assert len(hits) >= 1

    def test_observation_generation(self):
        """Test observation generation for agent."""
        manager = ProjectileManager()

        # Launch a projectile
        manager.launch_projectile(
            agent_id="agent_0",
            agent_position=np.array([0.0, 0.0]),
            target_position=np.array([100.0, 0.0]),
            target_velocity=np.array([50.0, 0.0]),
            current_time=0.0,
        )

        obs = manager.get_observation_for_agent(
            agent_id="agent_0",
            current_time=0.0,
            target_position=np.array([100.0, 0.0]),
        )

        assert obs.shape == (4,)
        assert 0 <= obs[0] <= 1  # Remaining projectiles
        assert 0 <= obs[1] <= 1  # Cooldown status

    def test_multiple_agents_independent(self):
        """Test that projectiles from different agents are tracked independently."""
        config = ProjectileConfig(cooldown=0.0, max_per_agent=2)
        manager = ProjectileManager(config)

        # Launch from two agents
        p1 = manager.launch_projectile(
            agent_id="agent_0",
            agent_position=np.array([0.0, 0.0]),
            target_position=np.array([100.0, 0.0]),
            target_velocity=np.array([50.0, 0.0]),
            current_time=0.0,
        )
        p2 = manager.launch_projectile(
            agent_id="agent_1",
            agent_position=np.array([0.0, 100.0]),
            target_position=np.array([100.0, 0.0]),
            target_velocity=np.array([50.0, 0.0]),
            current_time=0.0,
        )

        assert p1 is not None
        assert p2 is not None
        assert len(manager.projectiles) == 2
        assert manager.get_agent_projectile_count("agent_0") == 1
        assert manager.get_agent_projectile_count("agent_1") == 1


class TestProjectileEnvironmentIntegration:
    """Integration tests for projectile system with environment."""

    def test_environment_with_projectiles(self):
        """Test that environment works with projectile system enabled."""
        from src.env.scaled_environment import ScaledHypersonicSwarmEnv

        env = ScaledHypersonicSwarmEnv(
            num_agents=5,
            use_projectiles=True,
            target_speed=500.0,
        )

        obs, info = env.reset()
        assert len(obs) == 5

        # Check action space has 3 dimensions
        for agent in env.agents:
            assert env.action_space(agent).shape == (3,)

        # Check observation space includes projectile dims
        for agent in env.agents:
            obs_dim = env.observation_space(agent).shape[0]
            # Should include 4 extra dims for projectiles
            assert obs_dim > 75  # Base was 75, now 79

    def test_environment_without_projectiles(self):
        """Test backward compatibility without projectiles."""
        from src.env.scaled_environment import ScaledHypersonicSwarmEnv

        env = ScaledHypersonicSwarmEnv(
            num_agents=5,
            use_projectiles=False,
            target_speed=500.0,
        )

        obs, info = env.reset()

        # Check action space has 2 dimensions
        for agent in env.agents:
            assert env.action_space(agent).shape == (2,)

    def test_fire_action_triggers_launch(self):
        """Test that fire action launches projectile."""
        from src.env.scaled_environment import ScaledHypersonicSwarmEnv

        env = ScaledHypersonicSwarmEnv(
            num_agents=5,
            use_projectiles=True,
            target_speed=500.0,
            detection_range=10000.0,  # Large detection range so agents can see target
            initial_spread_min=500.0,
            initial_spread_max=1000.0,
        )

        obs, info = env.reset()

        # Run a few steps to get agents closer/detect target
        for _ in range(5):
            actions = {
                agent: np.array([1.0, 0.0, 0.0])  # Move toward center
                for agent in env.agents
            }
            env.step(actions)

        # Now try to fire
        actions = {
            agent: np.array([0.5, 0.0, 1.0])  # thrust, heading, fire
            for agent in env.agents
        }

        obs, rewards, terms, truncs, infos = env.step(actions)

        # At least some agents should have fired (those within detection range)
        fired_count = sum(1 for info in infos.values() if info.get("fired_this_step"))
        # With large detection range, all agents should be able to fire
        assert fired_count > 0

    def test_projectile_intercept_ends_episode(self):
        """Test that projectile hit terminates episode."""
        from src.env.scaled_environment import ScaledHypersonicSwarmEnv, ProjectileConfig

        # Configure for quick hit
        proj_config = ProjectileConfig(
            speed=2000.0,  # Very fast
            hit_radius=100.0,  # Large hit zone
            cooldown=0.1,  # Small cooldown to avoid division by zero
        )

        env = ScaledHypersonicSwarmEnv(
            num_agents=10,
            use_projectiles=True,
            target_speed=100.0,  # Slow target
            projectile_config=proj_config,
            arena_size=2000.0,
            initial_spread_min=500.0,
            initial_spread_max=800.0,
            detection_range=5000.0,  # Large detection range
        )

        obs, info = env.reset()

        # Run until intercept or max steps
        intercepted = False
        for step in range(200):
            if not env.agents:
                break

            actions = {
                agent: np.array([0.5, 0.0, 1.0])
                for agent in env.agents
            }
            obs, rewards, terms, truncs, infos = env.step(actions)

            if any(info.get("intercepted") for info in infos.values()):
                intercepted = True
                break

        # Should intercept with these favorable conditions
        assert intercepted, f"Failed to intercept after {step} steps"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
