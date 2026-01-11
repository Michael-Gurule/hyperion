"""Training module for HYPERION."""

from .train_marl import train_swarm_policy
from .curriculum import (
    CurriculumScheduler,
    ParameterizedCurriculumScheduler,
    ParameterizedCurriculumConfig,
    CurriculumStage,
    EvasionLevel,
    create_default_curriculum,
)
from .mappo import MAPPO, MAPPOConfig, MAPPOTrainer, ActorNetwork, CentralizedCritic
from .train_scaled import (
    ScaledTrainingConfig,
    train_with_mappo,
    train_with_rllib,
    AdaptiveCurriculumScheduler,
)
from .intrinsic_rewards import (
    IntrinsicRewardConfig,
    VelocityMismatchPenalty,
    InterceptGeometryBonus,
    NoveltySearch,
    SwarmCoverageBonus,
    IntrinsicRewardCalculator,
)

__all__ = [
    "train_swarm_policy",
    "CurriculumScheduler",
    "ParameterizedCurriculumScheduler",
    "ParameterizedCurriculumConfig",
    "CurriculumStage",
    "EvasionLevel",
    "create_default_curriculum",
    "MAPPO",
    "MAPPOConfig",
    "MAPPOTrainer",
    "ActorNetwork",
    "CentralizedCritic",
    "ScaledTrainingConfig",
    "train_with_mappo",
    "train_with_rllib",
    "AdaptiveCurriculumScheduler",
    "IntrinsicRewardConfig",
    "VelocityMismatchPenalty",
    "InterceptGeometryBonus",
    "NoveltySearch",
    "SwarmCoverageBonus",
    "IntrinsicRewardCalculator",
]
