"""
Projectile/Missile System for HYPERION.

Implements interceptor projectiles with Proportional Navigation (PN) guidance
to bridge the speed gap between slower drones and hypersonic targets.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class GuidanceType(Enum):
    """Projectile guidance modes."""
    PROPORTIONAL_NAVIGATION = "proportional_navigation"
    PURE_PURSUIT = "pure_pursuit"
    AUGMENTED_PN = "augmented_pn"


@dataclass
class ProjectileConfig:
    """Configuration for interceptor projectiles."""
    speed: float = 600.0  # units/s (faster than target)
    lifetime: float = 5.0  # seconds before fuel depletion
    hit_radius: float = 20.0  # impact zone radius
    cooldown: float = 2.0  # seconds between launches per agent
    max_per_agent: int = 3  # maximum projectiles per agent
    guidance_type: GuidanceType = GuidanceType.PROPORTIONAL_NAVIGATION
    nav_constant: float = 3.0  # PN gain (typically 3-5)
    max_acceleration: float = 200.0  # max lateral acceleration (units/s^2)
    max_turn_rate: float = 2.0  # rad/s projectile maneuverability


@dataclass
class Projectile:
    """Active projectile in the environment."""
    id: int
    owner_agent: str
    position: np.ndarray
    velocity: np.ndarray
    created_at: float
    active: bool = True

    # Tracking for PN guidance
    prev_los_angle: Optional[float] = None
    target_position_at_launch: Optional[np.ndarray] = None

    def copy(self) -> 'Projectile':
        """Create a copy of this projectile."""
        return Projectile(
            id=self.id,
            owner_agent=self.owner_agent,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            created_at=self.created_at,
            active=self.active,
            prev_los_angle=self.prev_los_angle,
            target_position_at_launch=self.target_position_at_launch.copy()
                if self.target_position_at_launch is not None else None,
        )


@dataclass
class ProjectileHit:
    """Information about a projectile hit."""
    projectile_id: int
    owner_agent: str
    hit_position: np.ndarray
    time: float


class ProportionalNavigationGuidance:
    """
    Proportional Navigation guidance law implementation.

    PN Law: a_cmd = N * V_c * (dLambda/dt)
    where:
        N = navigation constant (typically 3-5)
        V_c = closing velocity
        dLambda/dt = line-of-sight rate
    """

    def __init__(self, nav_constant: float = 3.0):
        """
        Initialize PN guidance.

        Args:
            nav_constant: Navigation constant N (typically 3-5)
        """
        self.N = nav_constant

    def compute_los_angle(
        self,
        projectile_pos: np.ndarray,
        target_pos: np.ndarray,
    ) -> float:
        """
        Compute line-of-sight angle from projectile to target.

        Args:
            projectile_pos: Projectile position [x, y]
            target_pos: Target position [x, y]

        Returns:
            LOS angle in radians
        """
        los_vector = target_pos - projectile_pos
        return np.arctan2(los_vector[1], los_vector[0])

    def compute_los_rate(
        self,
        current_los: float,
        prev_los: float,
        dt: float,
    ) -> float:
        """
        Compute line-of-sight rate.

        Args:
            current_los: Current LOS angle (radians)
            prev_los: Previous LOS angle (radians)
            dt: Time step

        Returns:
            LOS rate (rad/s)
        """
        # Handle angle wrapping
        delta = current_los - prev_los
        if delta > np.pi:
            delta -= 2 * np.pi
        elif delta < -np.pi:
            delta += 2 * np.pi

        return delta / dt if dt > 0 else 0.0

    def compute_closing_velocity(
        self,
        projectile_pos: np.ndarray,
        projectile_vel: np.ndarray,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
    ) -> float:
        """
        Compute closing velocity (rate of range decrease).

        Args:
            projectile_pos: Projectile position
            projectile_vel: Projectile velocity
            target_pos: Target position
            target_vel: Target velocity

        Returns:
            Closing velocity (positive when approaching)
        """
        # Relative position and velocity
        rel_pos = target_pos - projectile_pos
        rel_vel = target_vel - projectile_vel

        distance = np.linalg.norm(rel_pos)
        if distance < 1e-6:
            return 0.0

        # Closing velocity is negative of range rate
        los_unit = rel_pos / distance
        closing_vel = -np.dot(rel_vel, los_unit)

        return closing_vel

    def compute_acceleration_command(
        self,
        projectile_pos: np.ndarray,
        projectile_vel: np.ndarray,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
        prev_los_angle: Optional[float],
        dt: float,
    ) -> Tuple[np.ndarray, float]:
        """
        Compute PN guidance acceleration command.

        Args:
            projectile_pos: Projectile position
            projectile_vel: Projectile velocity
            target_pos: Target position
            target_vel: Target velocity
            prev_los_angle: Previous LOS angle (None on first call)
            dt: Time step

        Returns:
            Tuple of (acceleration command vector, current LOS angle)
        """
        # Compute current LOS angle
        current_los = self.compute_los_angle(projectile_pos, target_pos)

        # On first call, no LOS rate available
        if prev_los_angle is None:
            return np.array([0.0, 0.0]), current_los

        # Compute LOS rate
        los_rate = self.compute_los_rate(current_los, prev_los_angle, dt)

        # Compute closing velocity
        closing_vel = self.compute_closing_velocity(
            projectile_pos, projectile_vel, target_pos, target_vel
        )

        # PN acceleration magnitude
        accel_magnitude = self.N * closing_vel * los_rate

        # Acceleration is perpendicular to LOS (toward target motion)
        los_vector = target_pos - projectile_pos
        distance = np.linalg.norm(los_vector)

        if distance < 1e-6:
            return np.array([0.0, 0.0]), current_los

        # Perpendicular direction (rotate LOS by 90 degrees)
        los_unit = los_vector / distance
        perp_unit = np.array([-los_unit[1], los_unit[0]])

        # Sign based on LOS rate direction
        if los_rate < 0:
            perp_unit = -perp_unit

        accel_cmd = abs(accel_magnitude) * perp_unit

        return accel_cmd, current_los


class ProjectileManager:
    """Manages all active projectiles in the environment."""

    def __init__(self, config: Optional[ProjectileConfig] = None):
        """
        Initialize projectile manager.

        Args:
            config: Projectile configuration
        """
        self.config = config or ProjectileConfig()
        self.projectiles: List[Projectile] = []
        self.agent_cooldowns: Dict[str, float] = {}
        self.agent_projectile_counts: Dict[str, int] = {}
        self.next_projectile_id = 0
        self.hits: List[ProjectileHit] = []

        # Initialize PN guidance
        self.guidance = ProportionalNavigationGuidance(
            nav_constant=self.config.nav_constant
        )

    def reset(self):
        """Reset projectile manager state."""
        self.projectiles = []
        self.agent_cooldowns = {}
        self.agent_projectile_counts = {}
        self.next_projectile_id = 0
        self.hits = []

    def can_launch(self, agent_id: str, current_time: float) -> bool:
        """
        Check if agent can launch a projectile.

        Args:
            agent_id: Agent identifier
            current_time: Current simulation time

        Returns:
            True if launch is allowed
        """
        # Check cooldown
        if current_time < self.agent_cooldowns.get(agent_id, 0.0):
            return False

        # Check max projectiles
        active_count = self.agent_projectile_counts.get(agent_id, 0)
        if active_count >= self.config.max_per_agent:
            return False

        return True

    def get_cooldown_remaining(self, agent_id: str, current_time: float) -> float:
        """Get remaining cooldown time for an agent."""
        cooldown_end = self.agent_cooldowns.get(agent_id, 0.0)
        return max(0.0, cooldown_end - current_time)

    def get_agent_projectile_count(self, agent_id: str) -> int:
        """Get number of active projectiles for an agent."""
        return self.agent_projectile_counts.get(agent_id, 0)

    def _calculate_intercept_point(
        self,
        agent_position: np.ndarray,
        target_position: np.ndarray,
        target_velocity: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate predicted intercept point using lead pursuit.

        Args:
            agent_position: Launch position
            target_position: Current target position
            target_velocity: Target velocity

        Returns:
            Predicted intercept point
        """
        # Distance to target
        distance = np.linalg.norm(target_position - agent_position)

        # Estimated time to intercept
        target_speed = np.linalg.norm(target_velocity)
        relative_speed = self.config.speed - target_speed * 0.3  # Conservative estimate

        if relative_speed <= 0:
            # Intercept unlikely, aim directly at target
            return target_position

        time_to_intercept = distance / relative_speed

        # Predict target position
        intercept_point = target_position + target_velocity * time_to_intercept

        return intercept_point

    def launch_projectile(
        self,
        agent_id: str,
        agent_position: np.ndarray,
        target_position: np.ndarray,
        target_velocity: np.ndarray,
        current_time: float,
    ) -> Optional[Projectile]:
        """
        Launch a projectile toward predicted intercept point.

        Args:
            agent_id: Launching agent identifier
            agent_position: Agent position at launch
            target_position: Current target position
            target_velocity: Current target velocity
            current_time: Current simulation time

        Returns:
            Launched projectile or None if launch not allowed
        """
        if not self.can_launch(agent_id, current_time):
            return None

        # Calculate intercept point
        intercept_point = self._calculate_intercept_point(
            agent_position, target_position, target_velocity
        )

        # Launch direction toward intercept point
        direction = intercept_point - agent_position
        distance = np.linalg.norm(direction)

        if distance < 1e-6:
            # Target too close, aim directly
            direction = target_position - agent_position
            distance = np.linalg.norm(direction)
            if distance < 1e-6:
                return None

        direction = direction / distance

        # Create projectile
        projectile = Projectile(
            id=self.next_projectile_id,
            owner_agent=agent_id,
            position=agent_position.copy(),
            velocity=direction * self.config.speed,
            created_at=current_time,
            active=True,
            prev_los_angle=None,
            target_position_at_launch=target_position.copy(),
        )

        self.projectiles.append(projectile)
        self.next_projectile_id += 1

        # Update agent state
        self.agent_cooldowns[agent_id] = current_time + self.config.cooldown
        self.agent_projectile_counts[agent_id] = \
            self.agent_projectile_counts.get(agent_id, 0) + 1

        return projectile

    def _apply_guidance(
        self,
        projectile: Projectile,
        target_position: np.ndarray,
        target_velocity: np.ndarray,
        dt: float,
    ):
        """
        Apply PN guidance to update projectile velocity.

        Args:
            projectile: Projectile to update
            target_position: Current target position
            target_velocity: Current target velocity
            dt: Time step
        """
        if self.config.guidance_type == GuidanceType.PURE_PURSUIT:
            # Simple pursuit - always aim at target
            direction = target_position - projectile.position
            distance = np.linalg.norm(direction)
            if distance > 1e-6:
                direction = direction / distance
                projectile.velocity = direction * self.config.speed
            return

        # Proportional Navigation
        accel_cmd, current_los = self.guidance.compute_acceleration_command(
            projectile.position,
            projectile.velocity,
            target_position,
            target_velocity,
            projectile.prev_los_angle,
            dt,
        )

        # Store current LOS for next iteration
        projectile.prev_los_angle = current_los

        # Limit acceleration
        accel_magnitude = np.linalg.norm(accel_cmd)
        if accel_magnitude > self.config.max_acceleration:
            accel_cmd = accel_cmd / accel_magnitude * self.config.max_acceleration

        # Apply acceleration
        projectile.velocity += accel_cmd * dt

        # Maintain constant speed (projectile has finite fuel, modeled as constant thrust)
        current_speed = np.linalg.norm(projectile.velocity)
        if current_speed > 1e-6:
            projectile.velocity = projectile.velocity / current_speed * self.config.speed

    def update(
        self,
        dt: float,
        current_time: float,
        target_position: np.ndarray,
        target_velocity: np.ndarray,
    ) -> List[ProjectileHit]:
        """
        Update all projectiles and check for hits.

        Args:
            dt: Time step
            current_time: Current simulation time
            target_position: Current target position
            target_velocity: Current target velocity

        Returns:
            List of hits this timestep
        """
        hits_this_step = []

        for projectile in self.projectiles:
            if not projectile.active:
                continue

            # Check lifetime
            age = current_time - projectile.created_at
            if age > self.config.lifetime:
                projectile.active = False
                self.agent_projectile_counts[projectile.owner_agent] = \
                    max(0, self.agent_projectile_counts.get(projectile.owner_agent, 1) - 1)
                continue

            # Apply guidance
            self._apply_guidance(projectile, target_position, target_velocity, dt)

            # Update position
            projectile.position += projectile.velocity * dt

            # Check for hit
            distance_to_target = np.linalg.norm(
                projectile.position - target_position
            )

            if distance_to_target <= self.config.hit_radius:
                hit = ProjectileHit(
                    projectile_id=projectile.id,
                    owner_agent=projectile.owner_agent,
                    hit_position=projectile.position.copy(),
                    time=current_time,
                )
                hits_this_step.append(hit)
                self.hits.append(hit)
                projectile.active = False
                self.agent_projectile_counts[projectile.owner_agent] = \
                    max(0, self.agent_projectile_counts.get(projectile.owner_agent, 1) - 1)

        # Clean up inactive projectiles
        self.projectiles = [p for p in self.projectiles if p.active]

        return hits_this_step

    def get_projectile_states(self) -> List[Dict]:
        """
        Get states of all active projectiles.

        Returns:
            List of projectile state dictionaries
        """
        return [
            {
                "id": p.id,
                "owner": p.owner_agent,
                "position": p.position.copy(),
                "velocity": p.velocity.copy(),
                "age": 0.0,  # Would need current_time to compute
                "active": p.active,
            }
            for p in self.projectiles
            if p.active
        ]

    def get_nearest_projectile_to_target(
        self,
        target_position: np.ndarray,
    ) -> Optional[Dict]:
        """
        Get the nearest active projectile to target.

        Args:
            target_position: Target position

        Returns:
            Nearest projectile info or None
        """
        if not self.projectiles:
            return None

        min_distance = float('inf')
        nearest = None

        for p in self.projectiles:
            if not p.active:
                continue
            distance = np.linalg.norm(p.position - target_position)
            if distance < min_distance:
                min_distance = distance
                nearest = {
                    "id": p.id,
                    "owner": p.owner_agent,
                    "position": p.position.copy(),
                    "distance": distance,
                }

        return nearest

    def get_observation_for_agent(
        self,
        agent_id: str,
        current_time: float,
        target_position: np.ndarray,
        normalize_distance: float = 1000.0,
    ) -> np.ndarray:
        """
        Get projectile-related observation for an agent.

        Returns 4-dim observation:
        - remaining_projectiles (normalized 0-1)
        - cooldown_status (0 = ready, 1 = on cooldown)
        - nearest_projectile_x (normalized relative position)
        - nearest_projectile_y (normalized relative position)

        Args:
            agent_id: Agent identifier
            current_time: Current time
            target_position: Target position
            normalize_distance: Distance normalization factor

        Returns:
            4-dim observation array
        """
        obs = np.zeros(4, dtype=np.float32)

        # Remaining projectiles (normalized)
        remaining = self.config.max_per_agent - self.get_agent_projectile_count(agent_id)
        obs[0] = remaining / self.config.max_per_agent

        # Cooldown status
        cooldown_remaining = self.get_cooldown_remaining(agent_id, current_time)
        if self.config.cooldown > 0:
            obs[1] = min(1.0, cooldown_remaining / self.config.cooldown)
        else:
            obs[1] = 0.0  # No cooldown

        # Nearest projectile to target
        nearest = self.get_nearest_projectile_to_target(target_position)
        if nearest is not None:
            rel_pos = nearest["position"] - target_position
            obs[2] = np.clip(rel_pos[0] / normalize_distance, -1.0, 1.0)
            obs[3] = np.clip(rel_pos[1] / normalize_distance, -1.0, 1.0)

        return obs
