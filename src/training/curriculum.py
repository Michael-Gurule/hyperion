"""
Curriculum learning for progressive training difficulty.
Starts with easier scenarios and gradually increases complexity.

Includes:
- Basic CurriculumScheduler for episode-based progression
- ParameterizedCurriculumScheduler for speed/evasion progression
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class EvasionLevel(Enum):
    """Evasion difficulty levels for curriculum."""
    NONE = "none"           # No evasion (ballistic)
    BASIC = "basic"         # Simple weaving
    MEDIUM = "medium"       # Random jinking
    FULL = "full"           # Full pursuit evasion


@dataclass
class CurriculumStage:
    """Configuration for a single curriculum stage."""
    name: str
    speed_multiplier: float  # Target speed as multiple of agent speed
    evasion_level: EvasionLevel
    evasion_probability: float
    target_behavior: str  # Maps to TargetBehavior enum
    success_threshold: float = 0.7  # Success rate needed to advance
    min_episodes: int = 100  # Minimum episodes before can advance

    def get_target_speed(self, agent_max_speed: float) -> float:
        """Calculate target speed based on agent speed."""
        return agent_max_speed * self.speed_multiplier


@dataclass
class ParameterizedCurriculumConfig:
    """Configuration for parameterized curriculum learning."""
    agent_max_speed: float = 300.0
    advancement_threshold: float = 0.7
    regression_threshold: float = 0.3
    window_size: int = 100
    min_episodes_per_stage: int = 100
    allow_regression: bool = True


class CurriculumScheduler:
    """
    Manages curriculum learning stages.
    Progressively increases task difficulty as agents improve.
    """

    def __init__(self, stages: List[Dict]):
        """
        Initialize curriculum scheduler.

        Args:
            stages: List of curriculum stage configurations
                   Each stage has: name, target_speed, num_agents, duration_episodes
        """
        self.stages = stages
        self.current_stage_idx = 0
        self.episodes_in_current_stage = 0

    def get_current_stage(self) -> Dict:
        """Get current curriculum stage configuration."""
        if self.current_stage_idx >= len(self.stages):
            return self.stages[-1]  # Stay at final stage

        return self.stages[self.current_stage_idx]

    def should_advance(self) -> bool:
        """Check if should advance to next stage."""
        current_stage = self.get_current_stage()
        duration = current_stage.get("duration_episodes", float("inf"))

        return self.episodes_in_current_stage >= duration

    def advance_stage(self) -> bool:
        """
        Advance to next curriculum stage.

        Returns:
            True if advanced, False if already at final stage
        """
        if self.current_stage_idx < len(self.stages) - 1:
            self.current_stage_idx += 1
            self.episodes_in_current_stage = 0
            return True
        return False

    def record_episode(self):
        """Record that an episode has completed."""
        self.episodes_in_current_stage += 1

        if self.should_advance():
            advanced = self.advance_stage()
            if advanced:
                print(f"\n{'=' * 60}")
                print(
                    f"Advancing to curriculum stage: {self.get_current_stage()['name']}"
                )
                print(f"{'=' * 60}\n")

    def get_env_config(self) -> Dict:
        """Get environment configuration for current stage."""
        stage = self.get_current_stage()

        return {
            "target_speed": stage.get("target_speed", 1700.0),
            "num_agents": stage.get("num_agents", 5),
        }

    def reset(self):
        """Reset to first stage."""
        self.current_stage_idx = 0
        self.episodes_in_current_stage = 0


class ParameterizedCurriculumScheduler:
    """
    Advanced curriculum scheduler with parameterized speed and evasion progression.

    Stages are defined by:
    - Speed multiplier relative to agent max speed
    - Evasion level and probability
    - Target behavior pattern

    Progression based on moving average success rate.
    """

    # Default stages matching the plan:
    # Stage 1: 1.5x speed, no evasion (ballistic)
    # Stage 2: 2.0x speed, basic weaving
    # Stage 3: 3.0x speed, medium random jink
    # Stage 4: 4.0x speed, full pursuit evasion
    DEFAULT_STAGES = [
        CurriculumStage(
            name="stage_1_slow_ballistic",
            speed_multiplier=1.5,
            evasion_level=EvasionLevel.NONE,
            evasion_probability=0.0,
            target_behavior="BALLISTIC",
            success_threshold=0.8,
            min_episodes=100,
        ),
        CurriculumStage(
            name="stage_2_medium_weaving",
            speed_multiplier=2.0,
            evasion_level=EvasionLevel.BASIC,
            evasion_probability=0.3,
            target_behavior="WEAVING",
            success_threshold=0.7,
            min_episodes=150,
        ),
        CurriculumStage(
            name="stage_3_fast_jinking",
            speed_multiplier=3.0,
            evasion_level=EvasionLevel.MEDIUM,
            evasion_probability=0.5,
            target_behavior="RANDOM_JINK",
            success_threshold=0.6,
            min_episodes=200,
        ),
        CurriculumStage(
            name="stage_4_hypersonic_evasive",
            speed_multiplier=4.0,
            evasion_level=EvasionLevel.FULL,
            evasion_probability=0.7,
            target_behavior="PURSUIT_EVASION",
            success_threshold=0.5,
            min_episodes=300,
        ),
    ]

    def __init__(
        self,
        config: Optional[ParameterizedCurriculumConfig] = None,
        stages: Optional[List[CurriculumStage]] = None,
    ):
        """
        Initialize parameterized curriculum scheduler.

        Args:
            config: Curriculum configuration
            stages: Custom stage definitions (uses defaults if None)
        """
        self.config = config or ParameterizedCurriculumConfig()
        self.stages = stages or self.DEFAULT_STAGES

        self.current_stage_idx = 0
        self.episode_results: List[Dict[str, Any]] = []
        self.stage_history: List[Dict[str, Any]] = []
        self.episodes_in_current_stage = 0

    @property
    def current_stage(self) -> CurriculumStage:
        """Get the current curriculum stage."""
        return self.stages[self.current_stage_idx]

    @property
    def stage_name(self) -> str:
        """Get current stage name."""
        return self.current_stage.name

    @property
    def num_stages(self) -> int:
        """Get total number of stages."""
        return len(self.stages)

    @property
    def is_final_stage(self) -> bool:
        """Check if at final stage."""
        return self.current_stage_idx >= len(self.stages) - 1

    def get_env_config(self) -> Dict[str, Any]:
        """
        Get environment configuration for current stage.

        Returns:
            Dict with target_speed, evasion settings, and behavior
        """
        stage = self.current_stage
        target_speed = stage.get_target_speed(self.config.agent_max_speed)

        return {
            "target_speed": target_speed,
            "evasion_probability": stage.evasion_probability,
            "target_behavior": stage.target_behavior,
            "evasion_level": stage.evasion_level.value,
            # Adversarial config mapping
            "adversarial_enabled": stage.evasion_probability > 0,
            "evasive_maneuvers": stage.evasion_level != EvasionLevel.NONE,
        }

    def get_adversarial_config(self) -> Dict[str, Any]:
        """Get adversarial configuration for current stage."""
        stage = self.current_stage

        # Map evasion level to jink parameters
        jink_params = {
            EvasionLevel.NONE: {"jink_frequency": 0.0, "jink_magnitude": 0.0},
            EvasionLevel.BASIC: {"jink_frequency": 0.3, "jink_magnitude": 200.0},
            EvasionLevel.MEDIUM: {"jink_frequency": 0.5, "jink_magnitude": 400.0},
            EvasionLevel.FULL: {"jink_frequency": 0.8, "jink_magnitude": 600.0},
        }

        params = jink_params.get(stage.evasion_level, jink_params[EvasionLevel.NONE])

        return {
            "enabled": stage.evasion_probability > 0,
            "evasive_maneuvers": stage.evasion_level != EvasionLevel.NONE,
            "evasion_probability": stage.evasion_probability,
            **params,
        }

    def get_success_rate(self, window: Optional[int] = None) -> float:
        """
        Calculate moving average success rate.

        Args:
            window: Window size for moving average (uses config default if None)

        Returns:
            Success rate as float [0, 1]
        """
        if not self.episode_results:
            return 0.0

        window = window or self.config.window_size
        recent = self.episode_results[-window:]
        return np.mean([r["success"] for r in recent])

    def update(self, success: bool, episode_reward: float = 0.0) -> Dict[str, Any]:
        """
        Update curriculum with episode result.

        Args:
            success: Whether episode achieved interception
            episode_reward: Total episode reward

        Returns:
            Dict with update info including stage_changed flag
        """
        self.episode_results.append({
            "success": success,
            "reward": episode_reward,
            "stage": self.current_stage_idx,
            "episode": len(self.episode_results),
        })
        self.episodes_in_current_stage += 1

        result = {
            "stage_changed": False,
            "direction": None,
            "current_stage": self.current_stage_idx,
            "success_rate": self.get_success_rate(),
            "episodes_in_stage": self.episodes_in_current_stage,
        }

        # Check minimum episodes requirement
        if self.episodes_in_current_stage < self.current_stage.min_episodes:
            return result

        success_rate = self.get_success_rate()

        # Check for advancement
        if (
            success_rate >= self.current_stage.success_threshold
            and not self.is_final_stage
        ):
            self._advance_stage()
            result["stage_changed"] = True
            result["direction"] = "advance"
            result["current_stage"] = self.current_stage_idx
            result["new_stage_name"] = self.current_stage.name

        # Check for regression
        elif (
            self.config.allow_regression
            and success_rate <= self.config.regression_threshold
            and self.current_stage_idx > 0
            and self.episodes_in_current_stage >= self.config.min_episodes_per_stage
        ):
            self._regress_stage()
            result["stage_changed"] = True
            result["direction"] = "regress"
            result["current_stage"] = self.current_stage_idx
            result["new_stage_name"] = self.current_stage.name

        return result

    def _advance_stage(self):
        """Advance to the next curriculum stage."""
        if self.current_stage_idx < len(self.stages) - 1:
            self.stage_history.append({
                "episode": len(self.episode_results),
                "from_stage": self.current_stage_idx,
                "to_stage": self.current_stage_idx + 1,
                "direction": "advance",
                "success_rate": self.get_success_rate(),
            })
            self.current_stage_idx += 1
            self.episodes_in_current_stage = 0

    def _regress_stage(self):
        """Regress to the previous curriculum stage."""
        if self.current_stage_idx > 0:
            self.stage_history.append({
                "episode": len(self.episode_results),
                "from_stage": self.current_stage_idx,
                "to_stage": self.current_stage_idx - 1,
                "direction": "regress",
                "success_rate": self.get_success_rate(),
            })
            self.current_stage_idx -= 1
            self.episodes_in_current_stage = 0

    def force_advance(self) -> bool:
        """Force advancement regardless of success rate."""
        if not self.is_final_stage:
            self._advance_stage()
            return True
        return False

    def force_set_stage(self, stage_idx: int):
        """Force set to a specific stage."""
        if 0 <= stage_idx < len(self.stages):
            self.current_stage_idx = stage_idx
            self.episodes_in_current_stage = 0

    def get_stage_summary(self) -> Dict[str, Any]:
        """Get summary of current stage configuration."""
        stage = self.current_stage
        return {
            "stage_index": self.current_stage_idx,
            "stage_name": stage.name,
            "speed_multiplier": stage.speed_multiplier,
            "target_speed": stage.get_target_speed(self.config.agent_max_speed),
            "evasion_level": stage.evasion_level.value,
            "evasion_probability": stage.evasion_probability,
            "target_behavior": stage.target_behavior,
            "success_threshold": stage.success_threshold,
            "current_success_rate": self.get_success_rate(),
            "episodes_in_stage": self.episodes_in_current_stage,
            "min_episodes": stage.min_episodes,
            "is_final": self.is_final_stage,
        }

    def get_training_stats(self) -> Dict[str, Any]:
        """Get overall training statistics."""
        if not self.episode_results:
            return {"total_episodes": 0}

        return {
            "total_episodes": len(self.episode_results),
            "current_stage": self.current_stage_idx,
            "stages_completed": max(r["stage"] for r in self.episode_results),
            "overall_success_rate": np.mean([r["success"] for r in self.episode_results]),
            "recent_success_rate": self.get_success_rate(),
            "stage_transitions": len(self.stage_history),
            "advances": sum(1 for h in self.stage_history if h["direction"] == "advance"),
            "regressions": sum(1 for h in self.stage_history if h["direction"] == "regress"),
        }

    def reset(self):
        """Reset curriculum to initial state."""
        self.current_stage_idx = 0
        self.episode_results = []
        self.stage_history = []
        self.episodes_in_current_stage = 0


def create_default_curriculum(
    agent_max_speed: float = 300.0,
    advancement_threshold: float = 0.7,
) -> ParameterizedCurriculumScheduler:
    """
    Create a parameterized curriculum scheduler with default stages.

    Args:
        agent_max_speed: Maximum speed of agents (used to compute target speeds)
        advancement_threshold: Success rate threshold for stage advancement

    Returns:
        Configured ParameterizedCurriculumScheduler
    """
    config = ParameterizedCurriculumConfig(
        agent_max_speed=agent_max_speed,
        advancement_threshold=advancement_threshold,
    )
    return ParameterizedCurriculumScheduler(config=config)
