"""Environment module for HYPERION."""

from .hypersonic_swarm_env import HypersonicSwarmEnv, env
from .rllib_wrapper import RLlibHyperionEnv
from .scaled_environment import (
    ScaledHypersonicSwarmEnv,
    create_scaled_env,
    AdversarialConfig,
    RewardConfig,
    TargetBehavior,
    AgentRole,
)
from .projectile_system import (
    ProjectileManager,
    ProjectileConfig,
    Projectile,
    ProjectileHit,
    GuidanceType,
    ProportionalNavigationGuidance,
)

__all__ = [
    "HypersonicSwarmEnv",
    "env",
    "RLlibHyperionEnv",
    "ScaledHypersonicSwarmEnv",
    "create_scaled_env",
    "AdversarialConfig",
    "RewardConfig",
    "TargetBehavior",
    "AgentRole",
    "ProjectileManager",
    "ProjectileConfig",
    "Projectile",
    "ProjectileHit",
    "GuidanceType",
    "ProportionalNavigationGuidance",
]
