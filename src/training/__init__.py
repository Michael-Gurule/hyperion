"""Training module for HYPERION."""

from .train_marl import train_swarm_policy
from .curriculum import CurriculumScheduler
from .mappo import MAPPO, MAPPOConfig, MAPPOTrainer, ActorNetwork, CentralizedCritic
from .train_scaled import (
    ScaledTrainingConfig,
    train_with_mappo,
    train_with_rllib,
    AdaptiveCurriculumScheduler,
)

__all__ = [
    "train_swarm_policy",
    "CurriculumScheduler",
    "MAPPO",
    "MAPPOConfig",
    "MAPPOTrainer",
    "ActorNetwork",
    "CentralizedCritic",
    "ScaledTrainingConfig",
    "train_with_mappo",
    "train_with_rllib",
    "AdaptiveCurriculumScheduler",
]
