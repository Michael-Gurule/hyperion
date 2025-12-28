"""
Curriculum learning for progressive training difficulty.
Starts with easier scenarios and gradually increases complexity.
"""

from typing import Dict, List, Optional
import numpy as np


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
