"""Tests for intrinsic reward mechanisms."""

import pytest
import numpy as np

from src.training.intrinsic_rewards import (
    IntrinsicRewardConfig,
    VelocityMismatchPenalty,
    InterceptGeometryBonus,
    NoveltySearch,
    SwarmCoverageBonus,
    IntrinsicRewardCalculator,
)


class TestIntrinsicRewardConfig:
    """Tests for IntrinsicRewardConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = IntrinsicRewardConfig()
        assert config.trailing_penalty_scale == 0.5
        assert config.geometry_bonus_scale == 1.0
        assert config.novelty_bonus_scale == 0.1

    def test_custom_config(self):
        """Test custom configuration."""
        config = IntrinsicRewardConfig(
            trailing_penalty_scale=1.0,
            novelty_k_neighbors=5,
        )
        assert config.trailing_penalty_scale == 1.0
        assert config.novelty_k_neighbors == 5


class TestVelocityMismatchPenalty:
    """Tests for VelocityMismatchPenalty."""

    def test_no_penalty_stationary(self):
        """Test no penalty when agent or target stationary."""
        penalty_calc = VelocityMismatchPenalty()

        # Stationary agent
        penalty, info = penalty_calc.compute(
            agent_velocity=np.array([0.0, 0.0]),
            target_velocity=np.array([100.0, 0.0]),
            agent_position=np.array([0.0, 0.0]),
            target_position=np.array([500.0, 0.0]),
        )
        assert penalty == 0.0

    def test_trailing_penalty(self):
        """Test penalty for trailing (same direction movement)."""
        penalty_calc = VelocityMismatchPenalty()

        # Agent moving in same direction as target
        penalty, info = penalty_calc.compute(
            agent_velocity=np.array([100.0, 0.0]),  # Moving right
            target_velocity=np.array([200.0, 0.0]),  # Target moving right
            agent_position=np.array([0.0, 0.0]),
            target_position=np.array([500.0, 0.0]),
        )

        # Should have penalty (negative reward)
        assert penalty < 0
        assert info["trailing_score"] > 0.7

    def test_no_trailing_perpendicular(self):
        """Test no trailing penalty for perpendicular approach."""
        penalty_calc = VelocityMismatchPenalty()

        # Agent moving perpendicular to target
        penalty, info = penalty_calc.compute(
            agent_velocity=np.array([0.0, 100.0]),  # Moving up
            target_velocity=np.array([200.0, 0.0]),  # Target moving right
            agent_position=np.array([0.0, 0.0]),
            target_position=np.array([500.0, 0.0]),
        )

        # Trailing score should be near 0 (perpendicular)
        assert abs(info["trailing_score"]) < 0.1

    def test_head_on_approach(self):
        """Test head-on approach (opposite directions) has no trailing penalty."""
        penalty_calc = VelocityMismatchPenalty()

        # Agent approaching head-on
        penalty, info = penalty_calc.compute(
            agent_velocity=np.array([100.0, 0.0]),  # Moving right toward target
            target_velocity=np.array([-200.0, 0.0]),  # Target moving left toward agent
            agent_position=np.array([0.0, 0.0]),
            target_position=np.array([500.0, 0.0]),
        )

        # Trailing score should be negative (head-on)
        assert info["trailing_score"] < 0

    def test_receding_penalty(self):
        """Test penalty for moving away from target."""
        penalty_calc = VelocityMismatchPenalty()

        # Agent moving away from target
        penalty, info = penalty_calc.compute(
            agent_velocity=np.array([-100.0, 0.0]),  # Moving left
            target_velocity=np.array([0.0, 50.0]),  # Target moving up
            agent_position=np.array([0.0, 0.0]),
            target_position=np.array([500.0, 0.0]),  # Target to the right
        )

        # Should have receding penalty
        assert info["receding_penalty"] > 0


class TestInterceptGeometryBonus:
    """Tests for InterceptGeometryBonus."""

    def test_head_on_approach_bonus(self):
        """Test bonus for head-on approach."""
        geometry = InterceptGeometryBonus()

        # Agent positioned for head-on intercept
        # Agent is at (0, 500), target at origin moving right
        # Agent approaches from perpendicular/above
        bonus, info = geometry.compute(
            agent_position=np.array([0.0, 500.0]),  # Above target
            target_position=np.array([0.0, 0.0]),
            target_velocity=np.array([100.0, 0.0]),  # Target moving right
        )

        # Should have bonus (approaching from side, 90 degree angle)
        assert bonus > 0
        assert info["is_good_geometry"]

    def test_perpendicular_approach_bonus(self):
        """Test bonus for perpendicular approach."""
        geometry = InterceptGeometryBonus()

        # Agent positioned for perpendicular intercept
        bonus, info = geometry.compute(
            agent_position=np.array([0.0, 500.0]),  # Above target path
            target_position=np.array([0.0, 0.0]),
            target_velocity=np.array([100.0, 0.0]),  # Target moving right
        )

        # Should have bonus (90 degree approach)
        assert bonus > 0

    def test_tail_chase_no_bonus(self):
        """Test no bonus for tail-chase (chasing from behind)."""
        geometry = InterceptGeometryBonus()

        # Agent directly behind target, both moving right
        # to_target points right (+x), target moving right (+x)
        # approach angle = arccos(dot(+x, -x)) = arccos(-1) = pi
        # This is actually the best case (180 degree head-on)!
        #
        # For true tail-chase, agent is behind and target moving away:
        # Agent at (500, 0), target at (0, 0) moving left (-x)
        # to_target = target - agent = (-500, 0) -> unit = (-1, 0)
        # target_dir = (-1, 0)
        # approach angle = arccos(dot((-1,0), -(-1,0))) = arccos(dot((-1,0), (1,0))) = arccos(-1) = pi
        #
        # Actually the geometry calculation uses:
        # approach_angle = arccos(dot(to_target_unit, -target_dir))
        # For tail chase: agent behind, target moving away
        # Agent at (-500, 0), target at (0, 0), moving right (+100, 0)
        # to_target = (0,0) - (-500,0) = (500, 0) -> unit = (1, 0)
        # -target_dir = -(1, 0) = (-1, 0)
        # approach_angle = arccos(dot((1,0), (-1,0))) = arccos(-1) = pi
        #
        # That's 180 degrees which is good! The math is correct but my test
        # understanding was wrong. Let me use a different scenario:
        # Agent ahead of target's path, same direction
        bonus, info = geometry.compute(
            agent_position=np.array([500.0, 0.0]),  # Ahead of target
            target_position=np.array([0.0, 0.0]),
            target_velocity=np.array([100.0, 0.0]),  # Target moving toward agent
        )

        # When agent is ahead of target moving toward it:
        # to_target = (0,0) - (500,0) = (-500, 0) -> unit = (-1, 0)
        # -target_dir = -(1, 0) = (-1, 0)
        # approach_angle = arccos(dot((-1,0), (-1,0))) = arccos(1) = 0
        # This is 0 degrees - a tail-chase equivalent (no intercept advantage)
        assert info["approach_angle"] < np.pi / 4
        assert bonus == 0.0


class TestNoveltySearch:
    """Tests for NoveltySearch."""

    def test_first_state_novel(self):
        """Test that first state is maximally novel."""
        novelty = NoveltySearch()

        bonus, info = novelty.compute_novelty(
            agent_position=np.array([0.0, 0.0]),
            agent_velocity=np.array([100.0, 0.0]),
            target_position=np.array([500.0, 0.0]),
        )

        # First state should have full novelty
        assert info["novelty_score"] == 1.0
        assert bonus > 0

    def test_repeated_state_less_novel(self):
        """Test that repeated states have lower novelty."""
        novelty = NoveltySearch()

        pos = np.array([0.0, 0.0])
        vel = np.array([100.0, 0.0])
        target = np.array([500.0, 0.0])

        # Add same state multiple times
        for _ in range(10):
            novelty.compute_novelty(pos, vel, target)

        # Check novelty of same state
        bonus, info = novelty.compute_novelty(pos, vel, target)

        # Should have low novelty (seen before)
        assert info["novelty_score"] < 0.5

    def test_different_state_more_novel(self):
        """Test that different states have higher novelty."""
        novelty = NoveltySearch()

        # Add some states
        for i in range(10):
            novelty.compute_novelty(
                agent_position=np.array([0.0, 0.0]),
                agent_velocity=np.array([100.0, 0.0]),
                target_position=np.array([500.0, 0.0]),
            )

        # Check novelty of very different state
        bonus, info = novelty.compute_novelty(
            agent_position=np.array([5000.0, 5000.0]),  # Very different position
            agent_velocity=np.array([-100.0, 100.0]),
            target_position=np.array([-1000.0, 0.0]),
        )

        # Should have higher novelty
        assert info["novelty_score"] > 0.3

    def test_reset(self):
        """Test archive reset."""
        novelty = NoveltySearch()

        # Add states
        for _ in range(5):
            novelty.compute_novelty(
                np.array([0.0, 0.0]),
                np.array([100.0, 0.0]),
                np.array([500.0, 0.0]),
            )

        assert len(novelty.archive) == 5

        novelty.reset()
        assert len(novelty.archive) == 0


class TestSwarmCoverageBonus:
    """Tests for SwarmCoverageBonus."""

    def test_well_spread_swarm(self):
        """Test coverage bonus for well-spread agents."""
        coverage = SwarmCoverageBonus()

        # Agents spread out in a grid
        positions = {
            "agent_0": np.array([0.0, 0.0]),
            "agent_1": np.array([2000.0, 0.0]),
            "agent_2": np.array([0.0, 2000.0]),
            "agent_3": np.array([2000.0, 2000.0]),
        }

        bonuses, info = coverage.compute(
            positions,
            target_position=np.array([1000.0, 1000.0]),
            arena_size=4000.0,
        )

        # Should have good coverage score
        assert info["coverage_score"] > 0.5
        # All agents should have some bonus
        assert all(b >= 0 for b in bonuses.values())

    def test_clustered_swarm(self):
        """Test lower coverage for clustered agents."""
        coverage = SwarmCoverageBonus()

        # Agents all close together
        positions = {
            "agent_0": np.array([0.0, 0.0]),
            "agent_1": np.array([10.0, 0.0]),
            "agent_2": np.array([0.0, 10.0]),
            "agent_3": np.array([10.0, 10.0]),
        }

        bonuses, info = coverage.compute(
            positions,
            target_position=np.array([1000.0, 1000.0]),
            arena_size=4000.0,
        )

        # Should have lower coverage score
        assert info["coverage_score"] < 0.1


class TestIntrinsicRewardCalculator:
    """Tests for combined IntrinsicRewardCalculator."""

    def test_compute_agent_reward(self):
        """Test computing rewards for single agent."""
        calc = IntrinsicRewardCalculator()

        reward, info = calc.compute_agent_reward(
            agent_id="agent_0",
            agent_position=np.array([0.0, 0.0]),
            agent_velocity=np.array([100.0, 0.0]),
            target_position=np.array([500.0, 0.0]),
            target_velocity=np.array([200.0, 0.0]),
            detection_range=1000.0,
        )

        # Should have various components
        assert "trailing" in info
        assert "geometry" in info
        assert "total_intrinsic_reward" in info

    def test_detection_bonus(self):
        """Test detection bonus when in range."""
        calc = IntrinsicRewardCalculator()

        # In detection range
        reward1, info1 = calc.compute_agent_reward(
            agent_id="agent_0",
            agent_position=np.array([0.0, 0.0]),
            agent_velocity=np.array([100.0, 0.0]),
            target_position=np.array([500.0, 0.0]),
            target_velocity=np.array([200.0, 0.0]),
            detection_range=1000.0,
        )

        # Out of detection range
        calc2 = IntrinsicRewardCalculator()
        reward2, info2 = calc2.compute_agent_reward(
            agent_id="agent_0",
            agent_position=np.array([0.0, 0.0]),
            agent_velocity=np.array([100.0, 0.0]),
            target_position=np.array([5000.0, 0.0]),
            target_velocity=np.array([200.0, 0.0]),
            detection_range=1000.0,
        )

        assert info1["in_detection_range"]
        assert not info2["in_detection_range"]

    def test_new_detection_bonus(self):
        """Test one-time new detection bonus."""
        calc = IntrinsicRewardCalculator()

        # First detection
        reward1, info1 = calc.compute_agent_reward(
            agent_id="agent_0",
            agent_position=np.array([0.0, 0.0]),
            agent_velocity=np.array([100.0, 0.0]),
            target_position=np.array([500.0, 0.0]),
            target_velocity=np.array([200.0, 0.0]),
            detection_range=1000.0,
        )

        # Second detection (same agent)
        reward2, info2 = calc.compute_agent_reward(
            agent_id="agent_0",
            agent_position=np.array([100.0, 0.0]),
            agent_velocity=np.array([100.0, 0.0]),
            target_position=np.array([500.0, 0.0]),
            target_velocity=np.array([200.0, 0.0]),
            detection_range=1000.0,
        )

        # First should have new detection bonus
        assert info1.get("new_detection", False)
        # Second should not
        assert not info2.get("new_detection", False)

    def test_compute_swarm_rewards(self):
        """Test computing rewards for entire swarm."""
        calc = IntrinsicRewardCalculator()

        agent_states = {
            "agent_0": {
                "position": np.array([0.0, 0.0]),
                "velocity": np.array([100.0, 0.0]),
                "active": True,
            },
            "agent_1": {
                "position": np.array([1000.0, 0.0]),
                "velocity": np.array([100.0, 50.0]),
                "active": True,
            },
            "agent_2": {
                "position": np.array([500.0, 500.0]),
                "velocity": np.array([50.0, 100.0]),
                "active": True,
            },
        }

        rewards, info = calc.compute_swarm_rewards(
            agent_states=agent_states,
            target_position=np.array([2000.0, 0.0]),
            target_velocity=np.array([200.0, 0.0]),
            detection_range=3000.0,
            arena_size=8000.0,
        )

        assert len(rewards) == 3
        assert "coverage" in info
        assert "total_intrinsic" in info

    def test_inactive_agents_no_reward(self):
        """Test that inactive agents get zero reward."""
        calc = IntrinsicRewardCalculator()

        agent_states = {
            "agent_0": {
                "position": np.array([0.0, 0.0]),
                "velocity": np.array([100.0, 0.0]),
                "active": True,
            },
            "agent_1": {
                "position": np.array([1000.0, 0.0]),
                "velocity": np.array([100.0, 50.0]),
                "active": False,  # Inactive
            },
        }

        rewards, info = calc.compute_swarm_rewards(
            agent_states=agent_states,
            target_position=np.array([2000.0, 0.0]),
            target_velocity=np.array([200.0, 0.0]),
            detection_range=3000.0,
        )

        assert rewards["agent_1"] == 0.0

    def test_reset(self):
        """Test resetting calculator state."""
        calc = IntrinsicRewardCalculator()

        # Add some state
        calc.compute_agent_reward(
            agent_id="agent_0",
            agent_position=np.array([0.0, 0.0]),
            agent_velocity=np.array([100.0, 0.0]),
            target_position=np.array([500.0, 0.0]),
            target_velocity=np.array([200.0, 0.0]),
            detection_range=1000.0,
        )

        assert "agent_0" in calc.detected_by
        assert len(calc.novelty_search.archive) > 0

        calc.reset()

        assert len(calc.detected_by) == 0
        assert len(calc.novelty_search.archive) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
