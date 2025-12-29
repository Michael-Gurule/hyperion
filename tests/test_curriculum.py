"""Tests for curriculum learning schedulers."""

import pytest
import numpy as np

from src.training.curriculum import (
    CurriculumScheduler,
    ParameterizedCurriculumScheduler,
    ParameterizedCurriculumConfig,
    CurriculumStage,
    EvasionLevel,
    create_default_curriculum,
)


class TestCurriculumStage:
    """Tests for CurriculumStage dataclass."""

    def test_stage_creation(self):
        """Test basic stage creation."""
        stage = CurriculumStage(
            name="test_stage",
            speed_multiplier=2.0,
            evasion_level=EvasionLevel.BASIC,
            evasion_probability=0.3,
            target_behavior="WEAVING",
        )
        assert stage.name == "test_stage"
        assert stage.speed_multiplier == 2.0
        assert stage.evasion_level == EvasionLevel.BASIC

    def test_get_target_speed(self):
        """Test target speed calculation."""
        stage = CurriculumStage(
            name="test",
            speed_multiplier=2.5,
            evasion_level=EvasionLevel.NONE,
            evasion_probability=0.0,
            target_behavior="BALLISTIC",
        )

        # 2.5x agent speed
        assert stage.get_target_speed(300.0) == 750.0
        assert stage.get_target_speed(400.0) == 1000.0


class TestParameterizedCurriculumConfig:
    """Tests for ParameterizedCurriculumConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ParameterizedCurriculumConfig()
        assert config.agent_max_speed == 300.0
        assert config.advancement_threshold == 0.7
        assert config.regression_threshold == 0.3
        assert config.window_size == 100

    def test_custom_config(self):
        """Test custom configuration."""
        config = ParameterizedCurriculumConfig(
            agent_max_speed=400.0,
            advancement_threshold=0.8,
            window_size=50,
        )
        assert config.agent_max_speed == 400.0
        assert config.advancement_threshold == 0.8
        assert config.window_size == 50


class TestParameterizedCurriculumScheduler:
    """Tests for ParameterizedCurriculumScheduler."""

    def test_default_stages(self):
        """Test default stage configuration."""
        scheduler = ParameterizedCurriculumScheduler()

        # Should have 4 default stages
        assert scheduler.num_stages == 4

        # Check stage progression
        assert scheduler.stages[0].speed_multiplier == 1.5  # Stage 1
        assert scheduler.stages[1].speed_multiplier == 2.0  # Stage 2
        assert scheduler.stages[2].speed_multiplier == 3.0  # Stage 3
        assert scheduler.stages[3].speed_multiplier == 4.0  # Stage 4

    def test_evasion_progression(self):
        """Test evasion level progression across stages."""
        scheduler = ParameterizedCurriculumScheduler()

        assert scheduler.stages[0].evasion_level == EvasionLevel.NONE
        assert scheduler.stages[1].evasion_level == EvasionLevel.BASIC
        assert scheduler.stages[2].evasion_level == EvasionLevel.MEDIUM
        assert scheduler.stages[3].evasion_level == EvasionLevel.FULL

    def test_evasion_probability_progression(self):
        """Test evasion probability progression."""
        scheduler = ParameterizedCurriculumScheduler()

        assert scheduler.stages[0].evasion_probability == 0.0
        assert scheduler.stages[1].evasion_probability == 0.3
        assert scheduler.stages[2].evasion_probability == 0.5
        assert scheduler.stages[3].evasion_probability == 0.7

    def test_initial_state(self):
        """Test initial scheduler state."""
        scheduler = ParameterizedCurriculumScheduler()

        assert scheduler.current_stage_idx == 0
        assert scheduler.stage_name == "stage_1_slow_ballistic"
        assert not scheduler.is_final_stage

    def test_get_env_config(self):
        """Test environment configuration retrieval."""
        config = ParameterizedCurriculumConfig(agent_max_speed=300.0)
        scheduler = ParameterizedCurriculumScheduler(config=config)

        env_config = scheduler.get_env_config()

        # Stage 1: 1.5x speed = 450
        assert env_config["target_speed"] == 450.0
        assert env_config["evasion_probability"] == 0.0
        assert env_config["target_behavior"] == "BALLISTIC"
        assert not env_config["adversarial_enabled"]

    def test_get_adversarial_config(self):
        """Test adversarial configuration retrieval."""
        scheduler = ParameterizedCurriculumScheduler()

        # Stage 1 - no evasion
        adv_config = scheduler.get_adversarial_config()
        assert not adv_config["enabled"]
        assert adv_config["jink_frequency"] == 0.0

        # Force to stage 2
        scheduler.force_set_stage(1)
        adv_config = scheduler.get_adversarial_config()
        assert adv_config["enabled"]
        assert adv_config["jink_frequency"] == 0.3

    def test_success_rate_calculation(self):
        """Test moving average success rate calculation."""
        scheduler = ParameterizedCurriculumScheduler()

        # No episodes yet
        assert scheduler.get_success_rate() == 0.0

        # Add some results
        for _ in range(50):
            scheduler.update(success=True)
        for _ in range(50):
            scheduler.update(success=False)

        # Should be ~0.5 with window=100
        assert np.isclose(scheduler.get_success_rate(), 0.5, atol=0.01)

    def test_stage_advancement(self):
        """Test automatic stage advancement."""
        config = ParameterizedCurriculumConfig(window_size=10)
        scheduler = ParameterizedCurriculumScheduler(config=config)

        # Need min_episodes (100) before can advance
        # Simulate passing min episodes with high success
        for _ in range(100):
            scheduler.update(success=True)

        # Should have advanced from stage 0 to stage 1
        assert scheduler.current_stage_idx == 1
        assert scheduler.stage_name == "stage_2_medium_weaving"

    def test_stage_regression(self):
        """Test stage regression on poor performance."""
        config = ParameterizedCurriculumConfig(
            window_size=10,
            allow_regression=True,
            min_episodes_per_stage=10,
        )

        # Create custom stages with lower min_episodes for testing
        custom_stages = [
            CurriculumStage(
                name="stage_0",
                speed_multiplier=1.5,
                evasion_level=EvasionLevel.NONE,
                evasion_probability=0.0,
                target_behavior="BALLISTIC",
                success_threshold=0.7,
                min_episodes=5,
            ),
            CurriculumStage(
                name="stage_1",
                speed_multiplier=2.0,
                evasion_level=EvasionLevel.BASIC,
                evasion_probability=0.3,
                target_behavior="WEAVING",
                success_threshold=0.7,
                min_episodes=5,
            ),
        ]

        scheduler = ParameterizedCurriculumScheduler(
            config=config, stages=custom_stages
        )

        # First advance to stage 1
        scheduler.force_set_stage(1)

        # Simulate poor performance (need min_episodes_per_stage = 10)
        for _ in range(15):
            scheduler.update(success=False)

        # Should have regressed to stage 0
        assert scheduler.current_stage_idx == 0

    def test_no_regression_from_stage_0(self):
        """Test that cannot regress below stage 0."""
        config = ParameterizedCurriculumConfig(
            window_size=10,
            allow_regression=True,
            min_episodes_per_stage=5,
        )
        scheduler = ParameterizedCurriculumScheduler(config=config)

        # All failures at stage 0
        for _ in range(50):
            scheduler.update(success=False)

        # Should still be at stage 0
        assert scheduler.current_stage_idx == 0

    def test_final_stage_no_advancement(self):
        """Test cannot advance past final stage."""
        scheduler = ParameterizedCurriculumScheduler()

        # Go to final stage
        scheduler.force_set_stage(3)
        assert scheduler.is_final_stage

        # Try to force advance
        result = scheduler.force_advance()
        assert not result
        assert scheduler.current_stage_idx == 3

    def test_force_set_stage(self):
        """Test force setting to a specific stage."""
        scheduler = ParameterizedCurriculumScheduler()

        scheduler.force_set_stage(2)
        assert scheduler.current_stage_idx == 2
        assert scheduler.stage_name == "stage_3_fast_jinking"

        # Invalid stage should not change
        scheduler.force_set_stage(10)
        assert scheduler.current_stage_idx == 2

    def test_stage_summary(self):
        """Test stage summary generation."""
        config = ParameterizedCurriculumConfig(agent_max_speed=300.0)
        scheduler = ParameterizedCurriculumScheduler(config=config)

        summary = scheduler.get_stage_summary()

        assert summary["stage_index"] == 0
        assert summary["stage_name"] == "stage_1_slow_ballistic"
        assert summary["speed_multiplier"] == 1.5
        assert summary["target_speed"] == 450.0
        assert summary["evasion_level"] == "none"
        assert not summary["is_final"]

    def test_training_stats(self):
        """Test training statistics generation."""
        scheduler = ParameterizedCurriculumScheduler()

        # Empty stats
        stats = scheduler.get_training_stats()
        assert stats["total_episodes"] == 0

        # Add some episodes
        for _ in range(50):
            scheduler.update(success=True)
        for _ in range(50):
            scheduler.update(success=False)

        stats = scheduler.get_training_stats()
        assert stats["total_episodes"] == 100
        assert stats["overall_success_rate"] == 0.5

    def test_reset(self):
        """Test curriculum reset."""
        scheduler = ParameterizedCurriculumScheduler()

        # Advance and add history
        scheduler.force_set_stage(2)
        for _ in range(10):
            scheduler.update(success=True)

        # Reset
        scheduler.reset()

        assert scheduler.current_stage_idx == 0
        assert len(scheduler.episode_results) == 0
        assert len(scheduler.stage_history) == 0
        assert scheduler.episodes_in_current_stage == 0

    def test_min_episodes_requirement(self):
        """Test that min_episodes must be met before advancement."""
        config = ParameterizedCurriculumConfig(window_size=10)
        scheduler = ParameterizedCurriculumScheduler(config=config)

        # Stage 1 requires 100 min episodes
        # Add 50 successes - should not advance yet
        for _ in range(50):
            result = scheduler.update(success=True)
            assert not result["stage_changed"]

        # Still at stage 0
        assert scheduler.current_stage_idx == 0


class TestCreateDefaultCurriculum:
    """Tests for create_default_curriculum factory function."""

    def test_default_creation(self):
        """Test default curriculum creation."""
        curriculum = create_default_curriculum()

        assert curriculum.num_stages == 4
        assert curriculum.config.agent_max_speed == 300.0

    def test_custom_speed(self):
        """Test curriculum with custom agent speed."""
        curriculum = create_default_curriculum(agent_max_speed=400.0)

        # Stage 1: 1.5x = 600
        env_config = curriculum.get_env_config()
        assert env_config["target_speed"] == 600.0

    def test_custom_threshold(self):
        """Test curriculum with custom advancement threshold."""
        curriculum = create_default_curriculum(advancement_threshold=0.9)
        assert curriculum.config.advancement_threshold == 0.9


class TestEvasionLevel:
    """Tests for EvasionLevel enum."""

    def test_evasion_levels(self):
        """Test all evasion levels exist."""
        assert EvasionLevel.NONE.value == "none"
        assert EvasionLevel.BASIC.value == "basic"
        assert EvasionLevel.MEDIUM.value == "medium"
        assert EvasionLevel.FULL.value == "full"


class TestLegacyCurriculumScheduler:
    """Tests for legacy CurriculumScheduler (backward compatibility)."""

    def test_basic_operation(self):
        """Test basic curriculum scheduler operation."""
        stages = [
            {"name": "stage1", "target_speed": 500, "duration_episodes": 10},
            {"name": "stage2", "target_speed": 1000, "duration_episodes": 10},
        ]
        scheduler = CurriculumScheduler(stages)

        assert scheduler.get_current_stage()["name"] == "stage1"

        # Record episodes
        for _ in range(10):
            scheduler.record_episode()

        # Should have advanced
        assert scheduler.get_current_stage()["name"] == "stage2"

    def test_reset(self):
        """Test curriculum reset."""
        stages = [
            {"name": "stage1", "duration_episodes": 5},
            {"name": "stage2", "duration_episodes": 5},
        ]
        scheduler = CurriculumScheduler(stages)

        for _ in range(10):
            scheduler.record_episode()

        scheduler.reset()
        assert scheduler.current_stage_idx == 0
        assert scheduler.episodes_in_current_stage == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
