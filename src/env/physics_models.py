"""
Physics models for hypersonic threats and UAV dynamics.

Implements realistic atmospheric models, trajectory calculations,
and sensor physics for the HYPERION environment.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class AtmosphericConditions:
    """Atmospheric conditions at given altitude."""

    altitude: float  # meters
    temperature: float  # Kelvin
    pressure: float  # Pascals
    density: float  # kg/m^3
    speed_of_sound: float  # m/s


class AtmosphereModel:
    """
    Simplified atmospheric model (ISA - International Standard Atmosphere).
    Used for drag calculations and speed of sound.
    """

    # Sea level constants
    T0 = 288.15  # K
    P0 = 101325  # Pa
    RHO0 = 1.225  # kg/m^3

    # Gas constant for air
    R = 287.05  # J/(kg*K)
    GAMMA = 1.4  # Specific heat ratio

    # Temperature lapse rate (troposphere)
    LAPSE_RATE = -0.0065  # K/m

    @classmethod
    def get_conditions(cls, altitude: float) -> AtmosphericConditions:
        """
        Calculate atmospheric conditions at given altitude.

        Args:
            altitude: Altitude in meters (0-11000m for troposphere)

        Returns:
            Atmospheric conditions
        """
        # Limit to troposphere for simplicity
        altitude = np.clip(altitude, 0, 11000)

        # Temperature
        temperature = cls.T0 + cls.LAPSE_RATE * altitude

        # Pressure (barometric formula)
        pressure = cls.P0 * (temperature / cls.T0) ** (-9.81 / (cls.LAPSE_RATE * cls.R))

        # Density (ideal gas law)
        density = pressure / (cls.R * temperature)

        # Speed of sound
        speed_of_sound = np.sqrt(cls.GAMMA * cls.R * temperature)

        return AtmosphericConditions(
            altitude=altitude,
            temperature=temperature,
            pressure=pressure,
            density=density,
            speed_of_sound=speed_of_sound,
        )

    @classmethod
    def mach_to_velocity(cls, mach: float, altitude: float = 0.0) -> float:
        """Convert Mach number to velocity at given altitude."""
        conditions = cls.get_conditions(altitude)
        return mach * conditions.speed_of_sound

    @classmethod
    def velocity_to_mach(cls, velocity: float, altitude: float = 0.0) -> float:
        """Convert velocity to Mach number at given altitude."""
        conditions = cls.get_conditions(altitude)
        return velocity / conditions.speed_of_sound


class HypersonicTrajectory:
    """
    Hypersonic vehicle trajectory model.
    Implements realistic physics including atmospheric drag.
    """

    def __init__(
        self,
        initial_position: np.ndarray,
        initial_velocity: np.ndarray,
        mass: float = 1000.0,  # kg
        drag_coefficient: float = 0.3,
        reference_area: float = 0.5,  # m^2
    ):
        """
        Initialize hypersonic trajectory.

        Args:
            initial_position: Starting position [x, y] (meters)
            initial_velocity: Starting velocity [vx, vy] (m/s)
            mass: Vehicle mass (kg)
            drag_coefficient: Cd
            reference_area: Cross-sectional area (m^2)
        """
        self.position = initial_position.copy()
        self.velocity = initial_velocity.copy()
        self.mass = mass
        self.drag_coefficient = drag_coefficient
        self.reference_area = reference_area

    def calculate_drag_force(self, altitude: float = 0.0) -> np.ndarray:
        """
        Calculate atmospheric drag force.

        Args:
            altitude: Current altitude (meters)

        Returns:
            Drag force vector [fx, fy] (Newtons)
        """
        # Get atmospheric density
        conditions = AtmosphereModel.get_conditions(altitude)

        # Calculate speed
        speed = np.linalg.norm(self.velocity)

        if speed < 1e-6:
            return np.array([0.0, 0.0])

        # Drag force magnitude
        drag_magnitude = (
            0.5
            * conditions.density
            * speed**2
            * self.drag_coefficient
            * self.reference_area
        )

        # Drag force opposes velocity
        drag_direction = -self.velocity / speed

        return drag_magnitude * drag_direction

    def update(self, dt: float, altitude: float = 0.0):
        """
        Update trajectory using Euler integration.

        Args:
            dt: Time step (seconds)
            altitude: Current altitude (meters)
        """
        # Calculate drag force
        drag_force = self.calculate_drag_force(altitude)

        # Acceleration from drag
        acceleration = drag_force / self.mass

        # Update velocity
        self.velocity += acceleration * dt

        # Update position
        self.position += self.velocity * dt

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current position and velocity."""
        return self.position.copy(), self.velocity.copy()


class SensorModel:
    """
    Sensor models for detection and tracking.
    Includes noise, detection probability, and range effects.
    """

    @staticmethod
    def add_measurement_noise(measurement: np.ndarray, noise_std: float) -> np.ndarray:
        """
        Add Gaussian noise to measurement.

        Args:
            measurement: True measurement
            noise_std: Standard deviation of noise

        Returns:
            Noisy measurement
        """
        noise = np.random.normal(0, noise_std, measurement.shape)
        return measurement + noise

    @staticmethod
    def detection_probability(
        distance: float, max_range: float, base_probability: float = 0.95
    ) -> float:
        """
        Calculate probability of detection based on range.

        Args:
            distance: Distance to target (meters)
            max_range: Maximum sensor range (meters)
            base_probability: Detection probability at close range

        Returns:
            Detection probability [0-1]
        """
        if distance > max_range:
            return 0.0

        # Probability decreases with distance
        range_factor = 1.0 - (distance / max_range)
        probability = base_probability * range_factor**2

        return np.clip(probability, 0.0, 1.0)

    @staticmethod
    def is_detected(
        distance: float, max_range: float, base_probability: float = 0.95
    ) -> bool:
        """
        Determine if target is detected.

        Args:
            distance: Distance to target (meters)
            max_range: Maximum sensor range (meters)
            base_probability: Detection probability at close range

        Returns:
            True if detected
        """
        prob = SensorModel.detection_probability(distance, max_range, base_probability)
        return np.random.random() < prob


class EvasiveManeuvers:
    """
    Evasive maneuver models for hypersonic threats.
    Used to make interception more challenging.
    """

    @staticmethod
    def sinusoidal_weave(
        base_velocity: np.ndarray,
        time: float,
        amplitude: float = 100.0,
        frequency: float = 0.1,
    ) -> np.ndarray:
        """
        Add sinusoidal weaving to trajectory.

        Args:
            base_velocity: Nominal velocity vector
            time: Current time (seconds)
            amplitude: Weave amplitude (m/s)
            frequency: Weave frequency (Hz)

        Returns:
            Modified velocity vector
        """
        # Get perpendicular direction
        perpendicular = np.array([-base_velocity[1], base_velocity[0]])
        perpendicular = perpendicular / (np.linalg.norm(perpendicular) + 1e-6)

        # Add sinusoidal component
        offset = amplitude * np.sin(2 * np.pi * frequency * time)

        return base_velocity + offset * perpendicular

    @staticmethod
    def random_jink(
        base_velocity: np.ndarray, probability: float = 0.01, magnitude: float = 50.0
    ) -> np.ndarray:
        """
        Add random jinking maneuvers.

        Args:
            base_velocity: Nominal velocity vector
            probability: Probability of jink per timestep
            magnitude: Jink magnitude (m/s)

        Returns:
            Modified velocity vector
        """
        if np.random.random() < probability:
            # Random perpendicular offset
            angle = np.random.uniform(-np.pi / 4, np.pi / 4)
            rotation = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            return rotation @ base_velocity

        return base_velocity


def calculate_intercept_point(
    target_position: np.ndarray,
    target_velocity: np.ndarray,
    interceptor_position: np.ndarray,
    interceptor_speed: float,
) -> Optional[np.ndarray]:
    """
    Calculate optimal intercept point using proportional navigation.

    Args:
        target_position: Target position [x, y]
        target_velocity: Target velocity [vx, vy]
        interceptor_position: Interceptor position [x, y]
        interceptor_speed: Interceptor speed (scalar)

    Returns:
        Intercept point [x, y] or None if impossible
    """
    # Relative position
    relative_pos = target_position - interceptor_position

    # Time to intercept (simplified)
    target_speed = np.linalg.norm(target_velocity)

    if target_speed < 1e-6:
        return target_position

    # Calculate intercept time
    distance = np.linalg.norm(relative_pos)

    # Check if interception is possible
    if interceptor_speed <= target_speed * 0.9:
        # Intercept very difficult
        return None

    # Simplified intercept calculation
    time_to_intercept = distance / (interceptor_speed - target_speed * 0.5)

    # Predicted intercept point
    intercept_point = target_position + target_velocity * time_to_intercept

    return intercept_point
