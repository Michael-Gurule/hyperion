"""
Opportunistic Sensor Suite for Degraded Operations.

Implements multi-modal sensing using unconventional signal sources when
traditional radar/comms are unavailable or jammed. Enables continued
operation in contested electromagnetic environments.

Key innovations:
- Passive RF detection of hypersonic plasma sheath emissions
- Acoustic triangulation via sonic boom wavefront analysis
- Thermal wake detection from atmospheric heating
- Magnetic anomaly detection for metallic targets
- Swarm proprioception (inferring threat from neighbor behavior)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import warnings


class SensorStatus(Enum):
    """Operational status of a sensor."""
    OPERATIONAL = auto()
    DEGRADED = auto()
    JAMMED = auto()
    FAILED = auto()


class SignalType(Enum):
    """Types of signals that can be detected."""
    # Traditional
    RADAR_ACTIVE = "radar_active"
    RADAR_PASSIVE = "radar_passive"
    RF_COMMS = "rf_comms"

    # Unconventional/Opportunistic
    PLASMA_EMISSION = "plasma_emission"        # Hypersonic plasma sheath RF
    SONIC_BOOM = "sonic_boom"                  # Acoustic wavefront
    THERMAL_WAKE = "thermal_wake"              # Atmospheric heating
    MAGNETIC_ANOMALY = "magnetic_anomaly"      # Metallic mass detection
    OPTICAL_OCCLUSION = "optical_occlusion"    # Visual obstruction
    PRESSURE_WAVE = "pressure_wave"            # Atmospheric disturbance
    AMBIENT_RF_SCATTER = "ambient_rf_scatter"  # Passive coherent location
    CELESTIAL_NAV = "celestial_nav"            # Star tracker for positioning


@dataclass
class SensorReading:
    """Container for a single sensor measurement."""
    signal_type: SignalType
    timestamp: float

    # Position estimate (if available)
    position: Optional[np.ndarray] = None
    position_uncertainty: float = float('inf')

    # Velocity estimate (if available)
    velocity: Optional[np.ndarray] = None
    velocity_uncertainty: float = float('inf')

    # Raw signal strength and confidence
    signal_strength: float = 0.0
    confidence: float = 0.0

    # Sensor-specific metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Source agent (for distributed sensing)
    source_agent_id: Optional[str] = None


@dataclass
class SensorHealth:
    """Health and reliability metrics for a sensor."""
    status: SensorStatus = SensorStatus.OPERATIONAL
    reliability: float = 1.0  # [0, 1] learned reliability
    last_update: float = 0.0
    consecutive_failures: int = 0
    total_measurements: int = 0
    accepted_measurements: int = 0

    # Jamming detection
    jamming_detected: bool = False
    jamming_intensity: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        if self.total_measurements == 0:
            return 0.5
        return self.accepted_measurements / self.total_measurements


class BaseSensor(ABC):
    """Abstract base class for all sensors."""

    def __init__(
        self,
        signal_type: SignalType,
        max_range: float = 5000.0,
        noise_floor: float = 0.1,
        detection_threshold: float = 0.3,
    ):
        self.signal_type = signal_type
        self.max_range = max_range
        self.noise_floor = noise_floor
        self.detection_threshold = detection_threshold
        self.health = SensorHealth()

    @abstractmethod
    def detect(
        self,
        own_position: np.ndarray,
        own_velocity: np.ndarray,
        target_state: Optional[Dict] = None,
        environment: Optional[Dict] = None,
    ) -> Optional[SensorReading]:
        """Attempt to detect target using this sensor modality."""
        pass

    def apply_jamming(self, jamming_intensity: float):
        """Apply jamming effects to this sensor."""
        self.health.jamming_intensity = jamming_intensity
        self.health.jamming_detected = jamming_intensity > 0.3

        if jamming_intensity > 0.8:
            self.health.status = SensorStatus.JAMMED
        elif jamming_intensity > 0.4:
            self.health.status = SensorStatus.DEGRADED
        else:
            self.health.status = SensorStatus.OPERATIONAL

    def update_reliability(self, measurement_accepted: bool):
        """Update reliability estimate based on measurement acceptance."""
        self.health.total_measurements += 1
        if measurement_accepted:
            self.health.accepted_measurements += 1
            self.health.consecutive_failures = 0
            # Increase reliability slowly
            self.health.reliability = min(1.0, self.health.reliability + 0.01)
        else:
            self.health.consecutive_failures += 1
            # Decrease reliability faster on failures
            self.health.reliability = max(0.1, self.health.reliability - 0.05)

            if self.health.consecutive_failures > 5:
                self.health.status = SensorStatus.FAILED


class PlasmaEmissionDetector(BaseSensor):
    """
    Detects RF emissions from hypersonic plasma sheath.

    Physics: Objects traveling above Mach 5 create a plasma sheath due to
    atmospheric heating. This plasma emits broadband RF radiation that can
    be detected passively. The emission frequency and intensity correlate
    with velocity.

    Advantages:
    - Completely passive (no emissions to detect)
    - Works when active radar is jammed
    - Strongest signal at highest threat velocities

    Limitations:
    - Only works for hypersonic targets (>Mach 5)
    - Range limited by plasma emission strength
    - Affected by atmospheric conditions
    """

    def __init__(
        self,
        max_range: float = 10000.0,
        mach_threshold: float = 5.0,
        speed_of_sound: float = 343.0,
    ):
        super().__init__(
            signal_type=SignalType.PLASMA_EMISSION,
            max_range=max_range,
            noise_floor=0.05,
            detection_threshold=0.15,
        )
        self.mach_threshold = mach_threshold
        self.speed_of_sound = speed_of_sound

    def detect(
        self,
        own_position: np.ndarray,
        own_velocity: np.ndarray,
        target_state: Optional[Dict] = None,
        environment: Optional[Dict] = None,
    ) -> Optional[SensorReading]:
        if target_state is None:
            return None

        target_pos = target_state.get('position')
        target_vel = target_state.get('velocity')

        if target_pos is None or target_vel is None:
            return None

        # Check if target is hypersonic
        target_speed = np.linalg.norm(target_vel)
        mach_number = target_speed / self.speed_of_sound

        if mach_number < self.mach_threshold:
            return None  # No plasma sheath below Mach 5

        # Calculate distance
        distance = np.linalg.norm(target_pos - own_position)

        if distance > self.max_range:
            return None

        # Plasma emission strength increases with Mach number
        # Approximating blackbody radiation from plasma - increased baseline
        emission_strength = (mach_number - self.mach_threshold) ** 2 / 10.0 + 0.5

        # Signal decreases with distance squared
        signal_strength = emission_strength / (1.0 + (distance / 1000.0) ** 2)

        # Apply jamming degradation (plasma detection less affected by RF jamming)
        jamming_factor = 1.0 - 0.3 * self.health.jamming_intensity
        signal_strength *= jamming_factor

        # Add noise
        noise = np.random.normal(0, self.noise_floor)
        signal_strength = max(0, signal_strength + noise)

        if signal_strength < self.detection_threshold:
            return None

        # Position estimate from bearing (RF direction finding)
        # Add uncertainty based on signal strength
        bearing = np.arctan2(
            target_pos[1] - own_position[1],
            target_pos[0] - own_position[0]
        )
        bearing_noise = np.random.normal(0, 0.1 / signal_strength)
        estimated_bearing = bearing + bearing_noise

        # Range estimate from signal strength (inverse square law)
        estimated_range = 1000.0 * np.sqrt(emission_strength / signal_strength)
        range_uncertainty = estimated_range * 0.3  # 30% range uncertainty

        estimated_position = own_position + estimated_range * np.array([
            np.cos(estimated_bearing),
            np.sin(estimated_bearing)
        ])

        # Velocity estimate from Doppler shift of plasma emissions
        # Radial velocity component
        direction_to_target = (target_pos - own_position) / (distance + 1e-6)
        radial_velocity = np.dot(target_vel, direction_to_target)

        # Estimate full velocity (assume mostly radial for hypersonic)
        velocity_estimate = radial_velocity * direction_to_target

        confidence = min(1.0, signal_strength / self.detection_threshold)

        return SensorReading(
            signal_type=self.signal_type,
            timestamp=environment.get('time', 0.0) if environment else 0.0,
            position=estimated_position,
            position_uncertainty=range_uncertainty,
            velocity=velocity_estimate,
            velocity_uncertainty=target_speed * 0.2,  # 20% velocity uncertainty
            signal_strength=signal_strength,
            confidence=confidence * self.health.reliability,
            metadata={
                'mach_number': mach_number,
                'estimated_range': estimated_range,
                'bearing': estimated_bearing,
            }
        )


class AcousticArraySensor(BaseSensor):
    """
    Triangulates target position using sonic boom wavefront timing.

    Physics: Supersonic objects create a Mach cone. The arrival time of
    the sonic boom at different positions allows triangulation. Multiple
    drones act as a distributed acoustic array.

    Advantages:
    - Immune to RF jamming
    - Works in all weather
    - Can detect subsonic through hypersonic

    Limitations:
    - Requires multiple agents for triangulation
    - Delayed detection (speed of sound lag)
    - Affected by wind and atmospheric conditions
    """

    def __init__(
        self,
        max_range: float = 10000.0,
        speed_of_sound: float = 343.0,
        min_agents_for_triangulation: int = 3,
    ):
        super().__init__(
            signal_type=SignalType.SONIC_BOOM,
            max_range=max_range,
            noise_floor=0.05,
            detection_threshold=0.2,
        )
        self.speed_of_sound = speed_of_sound
        self.min_agents = min_agents_for_triangulation

        # Buffer for arrival times from swarm
        self.arrival_times: Dict[str, float] = {}
        self.arrival_positions: Dict[str, np.ndarray] = {}

    def record_arrival(
        self,
        agent_id: str,
        position: np.ndarray,
        arrival_time: float,
    ):
        """Record sonic boom arrival at an agent position."""
        self.arrival_times[agent_id] = arrival_time
        self.arrival_positions[agent_id] = position.copy()

    def detect(
        self,
        own_position: np.ndarray,
        own_velocity: np.ndarray,
        target_state: Optional[Dict] = None,
        environment: Optional[Dict] = None,
    ) -> Optional[SensorReading]:
        if target_state is None:
            return None

        target_pos = target_state.get('position')
        target_vel = target_state.get('velocity')
        current_time = environment.get('time', 0.0) if environment else 0.0

        if target_pos is None or target_vel is None:
            return None

        target_speed = np.linalg.norm(target_vel)

        # Must be supersonic to create sonic boom
        if target_speed < self.speed_of_sound:
            return None

        distance = np.linalg.norm(target_pos - own_position)

        if distance > self.max_range:
            return None

        # Calculate Mach angle
        mach_number = target_speed / self.speed_of_sound
        mach_angle = np.arcsin(1.0 / mach_number)

        # Time delay for sound to reach sensor
        sound_delay = distance / self.speed_of_sound

        # Simulate arrival time (in real system, this would be measured)
        arrival_time = current_time - sound_delay

        # Record this measurement
        self.record_arrival('self', own_position, arrival_time)

        # Check if we have enough agents for triangulation
        # In practice, this would use actual swarm data
        neighbor_positions = environment.get('neighbor_positions', []) if environment else []

        if len(neighbor_positions) < self.min_agents - 1:
            # Can't triangulate, return bearing-only estimate
            bearing = np.arctan2(
                target_pos[1] - own_position[1],
                target_pos[0] - own_position[0]
            )
            bearing_noise = np.random.normal(0, 0.2)

            # Very rough range estimate from sound intensity
            intensity = 1.0 / (1.0 + (distance / 2000.0) ** 2)
            estimated_range = 2000.0 / np.sqrt(intensity + 0.01)

            estimated_pos = own_position + estimated_range * np.array([
                np.cos(bearing + bearing_noise),
                np.sin(bearing + bearing_noise)
            ])

            return SensorReading(
                signal_type=self.signal_type,
                timestamp=current_time,
                position=estimated_pos,
                position_uncertainty=estimated_range * 0.5,  # 50% uncertainty
                velocity=None,  # Can't estimate velocity without triangulation
                confidence=0.3 * self.health.reliability,
                metadata={
                    'triangulated': False,
                    'mach_number': mach_number,
                    'num_agents': 1,
                }
            )

        # Multi-agent triangulation
        # Use TDOA (Time Difference of Arrival)
        positions = [own_position] + list(neighbor_positions[:self.min_agents-1])

        # Simulate arrival times at each position
        arrival_times = []
        for pos in positions:
            dist = np.linalg.norm(target_pos - pos)
            delay = dist / self.speed_of_sound
            noise = np.random.normal(0, 0.001)  # 1ms timing noise
            arrival_times.append(current_time - delay + noise)

        # TDOA triangulation (simplified)
        # In reality, would solve hyperbolic intersection
        estimated_pos = self._tdoa_triangulate(positions, arrival_times)

        # Velocity from Doppler shift of boom frequency
        # Simplified: estimate from direction and Mach number
        direction = (estimated_pos - own_position)
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        velocity_estimate = mach_number * self.speed_of_sound * direction

        position_uncertainty = 100.0  # Triangulation accuracy ~100m

        return SensorReading(
            signal_type=self.signal_type,
            timestamp=current_time,
            position=estimated_pos,
            position_uncertainty=position_uncertainty,
            velocity=velocity_estimate,
            velocity_uncertainty=target_speed * 0.15,
            signal_strength=1.0 / (1.0 + (distance / 2000.0)),
            confidence=0.7 * self.health.reliability,
            metadata={
                'triangulated': True,
                'mach_number': mach_number,
                'num_agents': len(positions),
                'tdoa_residual': 0.0,  # Would be computed in real implementation
            }
        )

    def _tdoa_triangulate(
        self,
        positions: List[np.ndarray],
        arrival_times: List[float],
    ) -> np.ndarray:
        """
        Triangulate position using Time Difference of Arrival.

        Simplified implementation - real system would use iterative
        least squares on hyperbolic equations.
        """
        if len(positions) < 3:
            return positions[0]

        # Reference position and time
        ref_pos = positions[0]
        ref_time = arrival_times[0]

        # Time differences
        tdoas = [t - ref_time for t in arrival_times[1:]]

        # Convert to range differences
        range_diffs = [tdoa * self.speed_of_sound for tdoa in tdoas]

        # Simplified: weighted average of positions
        # Real implementation would solve hyperbolic intersection
        weights = [1.0 / (abs(rd) + 100.0) for rd in range_diffs]
        total_weight = sum(weights)

        estimated_pos = ref_pos.copy()
        for pos, weight in zip(positions[1:], weights):
            estimated_pos += (pos - ref_pos) * weight / total_weight

        return estimated_pos


class ThermalWakeDetector(BaseSensor):
    """
    Detects thermal signatures from atmospheric heating.

    Physics: High-speed objects heat the surrounding air through compression
    and friction. This creates a thermal wake that can be detected by
    infrared sensors. The wake persists for several seconds.

    Advantages:
    - Passive detection
    - Can detect thermal wake even after target passes
    - Provides trajectory information

    Limitations:
    - Affected by atmospheric conditions
    - Background thermal noise
    - Limited range
    """

    def __init__(
        self,
        max_range: float = 4000.0,
        wake_persistence: float = 5.0,  # seconds
    ):
        super().__init__(
            signal_type=SignalType.THERMAL_WAKE,
            max_range=max_range,
            noise_floor=0.2,
            detection_threshold=0.3,
        )
        self.wake_persistence = wake_persistence

    def detect(
        self,
        own_position: np.ndarray,
        own_velocity: np.ndarray,
        target_state: Optional[Dict] = None,
        environment: Optional[Dict] = None,
    ) -> Optional[SensorReading]:
        if target_state is None:
            return None

        target_pos = target_state.get('position')
        target_vel = target_state.get('velocity')
        current_time = environment.get('time', 0.0) if environment else 0.0

        if target_pos is None or target_vel is None:
            return None

        distance = np.linalg.norm(target_pos - own_position)

        if distance > self.max_range:
            return None

        target_speed = np.linalg.norm(target_vel)

        # Thermal signature increases with speed^3 (kinetic heating)
        # Increased baseline for detection
        thermal_intensity = (target_speed / 1000.0) ** 3 + 0.3

        # Atmospheric absorption
        altitude = environment.get('altitude', 0.0) if environment else 0.0
        absorption = np.exp(-altitude / 10000.0)  # Scale height ~10km

        signal_strength = thermal_intensity * absorption / (1.0 + (distance / 1000.0) ** 2)

        # Apply jamming (thermal less affected than RF)
        jamming_factor = 1.0 - 0.2 * self.health.jamming_intensity
        signal_strength *= jamming_factor

        # Add thermal noise
        ambient_temp = environment.get('ambient_temperature', 250.0) if environment else 250.0
        thermal_noise = np.random.normal(0, self.noise_floor * (ambient_temp / 250.0))
        signal_strength = max(0, signal_strength + thermal_noise)

        if signal_strength < self.detection_threshold:
            return None

        # Position estimate from thermal centroid
        position_noise = np.random.normal(0, 200.0 / signal_strength, size=2)
        estimated_position = target_pos + position_noise

        # Velocity from thermal wake elongation direction
        wake_direction = target_vel / (target_speed + 1e-6)
        velocity_noise = np.random.normal(0, 0.1, size=2)
        velocity_estimate = target_vel + velocity_noise * target_speed

        confidence = min(1.0, signal_strength / self.detection_threshold)

        return SensorReading(
            signal_type=self.signal_type,
            timestamp=current_time,
            position=estimated_position,
            position_uncertainty=300.0 / signal_strength,
            velocity=velocity_estimate,
            velocity_uncertainty=target_speed * 0.25,
            signal_strength=signal_strength,
            confidence=confidence * self.health.reliability,
            metadata={
                'thermal_intensity': thermal_intensity,
                'wake_direction': wake_direction.tolist() if isinstance(wake_direction, np.ndarray) else wake_direction,
            }
        )


class MagneticAnomalyDetector(BaseSensor):
    """
    Detects metallic masses via magnetic field perturbations.

    Physics: Metallic objects perturb Earth's magnetic field. Sensitive
    magnetometers can detect these anomalies. Traditionally used for
    submarine detection, adapted here for aerial targets.

    Advantages:
    - Completely passive
    - Works through clouds/weather
    - Immune to RF jamming

    Limitations:
    - Very short range
    - Affected by local magnetic sources
    - Requires magnetically quiet platform
    """

    def __init__(
        self,
        max_range: float = 1000.0,  # MAD has limited range
        sensitivity: float = 0.1,  # nanoTesla
    ):
        super().__init__(
            signal_type=SignalType.MAGNETIC_ANOMALY,
            max_range=max_range,
            noise_floor=0.3,
            detection_threshold=0.4,
        )
        self.sensitivity = sensitivity

    def detect(
        self,
        own_position: np.ndarray,
        own_velocity: np.ndarray,
        target_state: Optional[Dict] = None,
        environment: Optional[Dict] = None,
    ) -> Optional[SensorReading]:
        if target_state is None:
            return None

        target_pos = target_state.get('position')
        target_mass = target_state.get('mass', 1000.0)  # kg
        current_time = environment.get('time', 0.0) if environment else 0.0

        if target_pos is None:
            return None

        distance = np.linalg.norm(target_pos - own_position)

        if distance > self.max_range:
            return None

        # Magnetic field falls off as 1/r^3 (dipole)
        # Anomaly strength proportional to magnetic moment (related to mass)
        magnetic_moment = target_mass * 0.01  # Simplified moment calculation

        # Field strength in nanoTesla
        field_strength = magnetic_moment / (distance ** 3 + 1.0)

        # Add magnetic noise (from platform, environment)
        own_speed = np.linalg.norm(own_velocity)
        motion_noise = own_speed * 0.001  # Motion induces noise
        environmental_noise = np.random.normal(0, self.noise_floor)

        signal_strength = field_strength / (self.sensitivity + motion_noise + abs(environmental_noise))

        if signal_strength < self.detection_threshold:
            return None

        # MAD provides poor positional accuracy but good detection
        bearing = np.arctan2(
            target_pos[1] - own_position[1],
            target_pos[0] - own_position[0]
        )
        bearing_noise = np.random.normal(0, 0.3)  # Poor bearing accuracy

        # Range estimate very uncertain
        estimated_range = (magnetic_moment / (signal_strength * self.sensitivity)) ** (1/3)

        estimated_position = own_position + estimated_range * np.array([
            np.cos(bearing + bearing_noise),
            np.sin(bearing + bearing_noise)
        ])

        confidence = min(1.0, signal_strength / self.detection_threshold) * 0.5  # Low confidence

        return SensorReading(
            signal_type=self.signal_type,
            timestamp=current_time,
            position=estimated_position,
            position_uncertainty=estimated_range * 0.5,  # 50% range uncertainty
            velocity=None,  # Cannot estimate velocity from MAD
            signal_strength=signal_strength,
            confidence=confidence * self.health.reliability,
            metadata={
                'field_strength_nT': field_strength,
                'estimated_mass': target_mass,
            }
        )


class PassiveCoherentLocation(BaseSensor):
    """
    Passive radar using ambient RF sources (cell towers, FM radio, etc.).

    Physics: Ambient RF transmitters (TV, radio, cellular) illuminate
    targets. By comparing direct signal with reflected signal, target
    position can be determined without own emissions.

    Advantages:
    - Completely passive (covert)
    - Uses existing infrastructure
    - Wide area coverage

    Limitations:
    - Requires knowledge of transmitter locations
    - Complex signal processing
    - Lower resolution than active radar
    """

    def __init__(
        self,
        max_range: float = 15000.0,
        reference_transmitters: Optional[List[Dict]] = None,
    ):
        super().__init__(
            signal_type=SignalType.AMBIENT_RF_SCATTER,
            max_range=max_range,
            noise_floor=0.25,
            detection_threshold=0.35,
        )
        # Default transmitter locations (would be real in production)
        self.transmitters = reference_transmitters or [
            {'position': np.array([0.0, 20000.0]), 'power': 50000.0, 'frequency': 100e6},
            {'position': np.array([15000.0, 0.0]), 'power': 100000.0, 'frequency': 900e6},
            {'position': np.array([-10000.0, -10000.0]), 'power': 25000.0, 'frequency': 88e6},
        ]

    def detect(
        self,
        own_position: np.ndarray,
        own_velocity: np.ndarray,
        target_state: Optional[Dict] = None,
        environment: Optional[Dict] = None,
    ) -> Optional[SensorReading]:
        if target_state is None:
            return None

        target_pos = target_state.get('position')
        target_vel = target_state.get('velocity')
        target_rcs = target_state.get('radar_cross_section', 1.0)  # m^2
        current_time = environment.get('time', 0.0) if environment else 0.0

        if target_pos is None:
            return None

        distance = np.linalg.norm(target_pos - own_position)

        if distance > self.max_range:
            return None

        # Calculate bistatic range for each transmitter
        best_signal = 0.0
        best_position_estimate = None

        for tx in self.transmitters:
            tx_pos = tx['position']
            tx_power = tx['power']

            # Bistatic range: transmitter -> target -> receiver
            tx_to_target = np.linalg.norm(target_pos - tx_pos)
            target_to_rx = distance

            # Bistatic radar equation (simplified)
            # Received power proportional to: Pt * G^2 * lambda^2 * RCS / ((4*pi)^3 * R_tx^2 * R_rx^2)
            wavelength = 3e8 / tx['frequency']
            signal_power = (tx_power * wavelength ** 2 * target_rcs /
                          ((4 * np.pi) ** 3 * tx_to_target ** 2 * target_to_rx ** 2 + 1e-6))

            if signal_power > best_signal:
                best_signal = signal_power

                # Position estimate from bistatic ellipse intersection
                # Simplified: weighted average based on geometry
                baseline = tx_pos - own_position
                baseline_len = np.linalg.norm(baseline)

                if baseline_len > 0:
                    # Estimate along perpendicular bisector
                    bistatic_range = tx_to_target + target_to_rx
                    perp = np.array([-baseline[1], baseline[0]]) / baseline_len

                    midpoint = (tx_pos + own_position) / 2
                    offset = np.sqrt(max(0, (bistatic_range / 2) ** 2 - (baseline_len / 2) ** 2))

                    # Choose side based on actual target position (in real system, would use multiple TXs)
                    if np.dot(target_pos - midpoint, perp) > 0:
                        best_position_estimate = midpoint + offset * perp
                    else:
                        best_position_estimate = midpoint - offset * perp

        if best_signal < self.detection_threshold * 1e-15:  # Threshold in power units
            return None

        # Apply jamming effects
        jamming_factor = 1.0 - 0.7 * self.health.jamming_intensity  # PCL affected by jamming
        best_signal *= jamming_factor

        if best_position_estimate is None:
            return None

        # Add noise to position estimate
        position_noise = np.random.normal(0, 500.0, size=2)  # 500m accuracy
        estimated_position = best_position_estimate + position_noise

        # Velocity from Doppler (if available)
        velocity_estimate = None
        if target_vel is not None:
            doppler_accuracy = 0.2
            velocity_noise = np.random.normal(0, doppler_accuracy, size=2) * np.linalg.norm(target_vel)
            velocity_estimate = target_vel + velocity_noise

        signal_strength = np.log10(best_signal + 1e-20) + 20  # dB scale, normalized
        signal_strength = max(0, min(1, (signal_strength + 150) / 50))  # Normalize to [0,1]

        confidence = signal_strength * 0.6  # PCL has moderate confidence

        return SensorReading(
            signal_type=self.signal_type,
            timestamp=current_time,
            position=estimated_position,
            position_uncertainty=500.0,
            velocity=velocity_estimate,
            velocity_uncertainty=100.0 if velocity_estimate is not None else float('inf'),
            signal_strength=signal_strength,
            confidence=confidence * self.health.reliability,
            metadata={
                'num_transmitters_used': len(self.transmitters),
                'bistatic_mode': True,
            }
        )


class SwarmProprioception(BaseSensor):
    """
    Infers threat location from observing neighbor drone behavior.

    This is the most novel sensor - it doesn't detect the target directly,
    but infers its location by observing how other swarm members react.

    Physics: If neighboring drones are accelerating toward a point,
    that point likely contains a threat. This works even under total
    communication blackout - just requires visual tracking of neighbors.

    Advantages:
    - Works under complete jamming
    - No emissions required
    - Leverages swarm intelligence
    - Emergent behavior detection

    Limitations:
    - Requires visible neighbors
    - Delayed inference
    - Can be fooled by coordinated decoys
    """

    def __init__(
        self,
        observation_range: float = 2000.0,
        min_neighbors_for_inference: int = 2,
        acceleration_threshold: float = 5.0,  # m/s^2
    ):
        super().__init__(
            signal_type=SignalType.OPTICAL_OCCLUSION,  # Visual observation
            max_range=observation_range,
            noise_floor=0.1,
            detection_threshold=0.2,
        )
        self.min_neighbors = min_neighbors_for_inference
        self.acceleration_threshold = acceleration_threshold

        # History of neighbor observations
        self.neighbor_history: Dict[str, List[Dict]] = {}
        self.inference_confidence = 0.0

    def observe_neighbors(
        self,
        neighbor_states: Dict[str, Dict],
        current_time: float,
    ):
        """
        Record neighbor states for inference.

        Args:
            neighbor_states: Dict of neighbor_id -> {position, velocity}
            current_time: Current timestamp
        """
        for neighbor_id, state in neighbor_states.items():
            if neighbor_id not in self.neighbor_history:
                self.neighbor_history[neighbor_id] = []

            self.neighbor_history[neighbor_id].append({
                'time': current_time,
                'position': state['position'].copy(),
                'velocity': state['velocity'].copy(),
            })

            # Keep only recent history (last 2 seconds)
            self.neighbor_history[neighbor_id] = [
                h for h in self.neighbor_history[neighbor_id]
                if current_time - h['time'] < 2.0
            ]

    def detect(
        self,
        own_position: np.ndarray,
        own_velocity: np.ndarray,
        target_state: Optional[Dict] = None,  # Not used - inferred from neighbors
        environment: Optional[Dict] = None,
    ) -> Optional[SensorReading]:
        """
        Infer target location from neighbor behavior.
        """
        current_time = environment.get('time', 0.0) if environment else 0.0
        neighbor_states = environment.get('neighbor_states', {}) if environment else {}

        # Update neighbor observations
        self.observe_neighbors(neighbor_states, current_time)

        # Need minimum neighbors for inference
        active_neighbors = [
            nid for nid, hist in self.neighbor_history.items()
            if len(hist) >= 2 and current_time - hist[-1]['time'] < 0.5
        ]

        if len(active_neighbors) < self.min_neighbors:
            return None

        # Compute acceleration vectors for each neighbor
        acceleration_vectors = []
        neighbor_positions = []

        for nid in active_neighbors:
            hist = self.neighbor_history[nid]
            if len(hist) < 2:
                continue

            # Compute acceleration from velocity change
            recent = hist[-1]
            older = hist[-2]

            dt = recent['time'] - older['time']
            if dt < 0.01:
                continue

            acceleration = (recent['velocity'] - older['velocity']) / dt
            accel_magnitude = np.linalg.norm(acceleration)

            if accel_magnitude > self.acceleration_threshold:
                # This neighbor is maneuvering significantly
                acceleration_vectors.append(acceleration)
                neighbor_positions.append(recent['position'])

        if len(acceleration_vectors) < self.min_neighbors:
            return None

        # Find convergence point of acceleration vectors
        # Each neighbor's acceleration points roughly toward threat

        # Use least squares to find intersection of acceleration lines
        estimated_target_pos = self._find_convergence_point(
            neighbor_positions,
            acceleration_vectors,
        )

        if estimated_target_pos is None:
            return None

        # Check if convergence point is reasonable
        distance_to_estimate = np.linalg.norm(estimated_target_pos - own_position)

        if distance_to_estimate > self.max_range * 2:
            return None  # Unreasonable estimate

        # Confidence based on convergence quality
        convergence_quality = self._compute_convergence_quality(
            neighbor_positions,
            acceleration_vectors,
            estimated_target_pos,
        )

        self.inference_confidence = convergence_quality

        if convergence_quality < self.detection_threshold:
            return None

        # Position uncertainty based on convergence
        position_uncertainty = 500.0 / (convergence_quality + 0.1)

        # Cannot estimate velocity from proprioception alone
        # But can estimate direction of motion from acceleration pattern
        mean_accel = np.mean(acceleration_vectors, axis=0)
        inferred_direction = mean_accel / (np.linalg.norm(mean_accel) + 1e-6)

        # Rough velocity estimate (assume target moving toward swarm centroid)
        swarm_centroid = np.mean(neighbor_positions, axis=0)
        velocity_direction = (swarm_centroid - estimated_target_pos)
        velocity_direction = velocity_direction / (np.linalg.norm(velocity_direction) + 1e-6)

        # Assume hypersonic if neighbors reacting strongly
        mean_accel_mag = np.mean([np.linalg.norm(a) for a in acceleration_vectors])
        inferred_speed = 500.0 + mean_accel_mag * 50  # Rough heuristic

        velocity_estimate = inferred_speed * velocity_direction

        return SensorReading(
            signal_type=SignalType.OPTICAL_OCCLUSION,
            timestamp=current_time,
            position=estimated_target_pos,
            position_uncertainty=position_uncertainty,
            velocity=velocity_estimate,
            velocity_uncertainty=inferred_speed * 0.5,  # High uncertainty
            signal_strength=convergence_quality,
            confidence=convergence_quality * self.health.reliability * 0.7,
            metadata={
                'inference_type': 'swarm_proprioception',
                'num_neighbors_used': len(acceleration_vectors),
                'convergence_quality': convergence_quality,
                'mean_acceleration': mean_accel_mag,
            }
        )

    def _find_convergence_point(
        self,
        positions: List[np.ndarray],
        directions: List[np.ndarray],
    ) -> Optional[np.ndarray]:
        """
        Find point where acceleration vectors converge.
        Uses least squares on line intersection.
        """
        if len(positions) < 2:
            return None

        # Build system of equations for line intersections
        # Each acceleration vector defines a ray from neighbor position

        A = []
        b = []

        for pos, direction in zip(positions, directions):
            dir_norm = direction / (np.linalg.norm(direction) + 1e-6)

            # Perpendicular to direction
            perp = np.array([-dir_norm[1], dir_norm[0]])

            # Point on line closest to origin when moving perpendicular
            A.append(perp)
            b.append(np.dot(perp, pos))

        A = np.array(A)
        b = np.array(b)

        # Least squares solution
        try:
            result, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            return result
        except np.linalg.LinAlgError:
            # Fallback: average of positions shifted by average direction
            avg_pos = np.mean(positions, axis=0)
            avg_dir = np.mean(directions, axis=0)
            avg_dir_norm = avg_dir / (np.linalg.norm(avg_dir) + 1e-6)

            # Project forward
            return avg_pos + avg_dir_norm * 1000.0

    def _compute_convergence_quality(
        self,
        positions: List[np.ndarray],
        directions: List[np.ndarray],
        target_point: np.ndarray,
    ) -> float:
        """
        Compute how well the acceleration vectors converge on the target point.
        """
        if len(positions) == 0:
            return 0.0

        alignment_scores = []

        for pos, direction in zip(positions, directions):
            # Vector from neighbor to estimated target
            to_target = target_point - pos
            to_target_norm = to_target / (np.linalg.norm(to_target) + 1e-6)

            # Normalize acceleration direction
            dir_norm = direction / (np.linalg.norm(direction) + 1e-6)

            # Dot product gives alignment (-1 to 1)
            alignment = np.dot(dir_norm, to_target_norm)

            # Convert to 0-1 score (only positive alignment counts)
            alignment_scores.append(max(0, alignment))

        # Average alignment
        mean_alignment = np.mean(alignment_scores)

        # Bonus for multiple agreeing neighbors
        agreement_bonus = min(1.0, len(positions) / 5.0)

        return mean_alignment * agreement_bonus


class OpportunisticSensorSuite:
    """
    Complete suite of opportunistic sensors for degraded operations.

    Manages multiple sensor modalities and provides unified interface
    for multi-modal sensing under jamming and communication denial.
    """

    def __init__(
        self,
        enable_plasma: bool = True,
        enable_acoustic: bool = True,
        enable_thermal: bool = True,
        enable_magnetic: bool = True,
        enable_pcl: bool = True,
        enable_proprioception: bool = True,
    ):
        """
        Initialize sensor suite with configurable modalities.
        """
        self.sensors: Dict[SignalType, BaseSensor] = {}

        if enable_plasma:
            self.sensors[SignalType.PLASMA_EMISSION] = PlasmaEmissionDetector()

        if enable_acoustic:
            self.sensors[SignalType.SONIC_BOOM] = AcousticArraySensor()

        if enable_thermal:
            self.sensors[SignalType.THERMAL_WAKE] = ThermalWakeDetector()

        if enable_magnetic:
            self.sensors[SignalType.MAGNETIC_ANOMALY] = MagneticAnomalyDetector()

        if enable_pcl:
            self.sensors[SignalType.AMBIENT_RF_SCATTER] = PassiveCoherentLocation()

        if enable_proprioception:
            self.sensors[SignalType.OPTICAL_OCCLUSION] = SwarmProprioception()

        # Global jamming state
        self.jamming_intensity = 0.0
        self.comms_degradation = 0.0

    def set_jamming_state(
        self,
        rf_jamming: float = 0.0,
        gps_jamming: float = 0.0,
        comms_degradation: float = 0.0,
    ):
        """
        Set current electromagnetic environment state.

        Args:
            rf_jamming: RF jamming intensity [0, 1]
            gps_jamming: GPS jamming intensity [0, 1]
            comms_degradation: Communication degradation [0, 1]
        """
        self.jamming_intensity = rf_jamming
        self.comms_degradation = comms_degradation

        # Apply jamming to each sensor based on susceptibility
        jamming_susceptibility = {
            SignalType.PLASMA_EMISSION: 0.3,      # Passive, less affected
            SignalType.SONIC_BOOM: 0.0,           # Acoustic, immune
            SignalType.THERMAL_WAKE: 0.1,         # IR, mostly immune
            SignalType.MAGNETIC_ANOMALY: 0.0,     # Magnetic, immune
            SignalType.AMBIENT_RF_SCATTER: 0.7,   # PCL, moderately affected
            SignalType.OPTICAL_OCCLUSION: 0.0,    # Visual, immune
        }

        for signal_type, sensor in self.sensors.items():
            susceptibility = jamming_susceptibility.get(signal_type, 0.5)
            effective_jamming = rf_jamming * susceptibility
            sensor.apply_jamming(effective_jamming)

    def get_all_readings(
        self,
        own_position: np.ndarray,
        own_velocity: np.ndarray,
        target_state: Optional[Dict] = None,
        environment: Optional[Dict] = None,
    ) -> List[SensorReading]:
        """
        Get readings from all available sensors.

        Returns list of readings from sensors that detect the target.
        """
        readings = []

        for signal_type, sensor in self.sensors.items():
            if sensor.health.status == SensorStatus.FAILED:
                continue

            try:
                reading = sensor.detect(
                    own_position=own_position,
                    own_velocity=own_velocity,
                    target_state=target_state,
                    environment=environment,
                )

                if reading is not None:
                    readings.append(reading)

            except Exception as e:
                warnings.warn(f"Sensor {signal_type} failed: {e}")
                sensor.health.consecutive_failures += 1

        return readings

    def get_best_reading(
        self,
        own_position: np.ndarray,
        own_velocity: np.ndarray,
        target_state: Optional[Dict] = None,
        environment: Optional[Dict] = None,
    ) -> Optional[SensorReading]:
        """
        Get single best reading across all sensors.
        """
        readings = self.get_all_readings(
            own_position=own_position,
            own_velocity=own_velocity,
            target_state=target_state,
            environment=environment,
        )

        if not readings:
            return None

        # Return highest confidence reading
        return max(readings, key=lambda r: r.confidence)

    def get_sensor_status(self) -> Dict[str, Dict]:
        """Get status of all sensors."""
        status = {}

        for signal_type, sensor in self.sensors.items():
            status[signal_type.value] = {
                'status': sensor.health.status.name,
                'reliability': sensor.health.reliability,
                'acceptance_rate': sensor.health.acceptance_rate,
                'jamming_detected': sensor.health.jamming_detected,
                'jamming_intensity': sensor.health.jamming_intensity,
            }

        return status

    def update_reliability(
        self,
        signal_type: SignalType,
        measurement_accepted: bool,
    ):
        """Update reliability for a specific sensor."""
        if signal_type in self.sensors:
            self.sensors[signal_type].update_reliability(measurement_accepted)

    def get_operational_sensors(self) -> List[SignalType]:
        """Get list of currently operational sensors."""
        return [
            st for st, sensor in self.sensors.items()
            if sensor.health.status in [SensorStatus.OPERATIONAL, SensorStatus.DEGRADED]
        ]

    def reset(self):
        """Reset all sensors to initial state."""
        for sensor in self.sensors.values():
            sensor.health = SensorHealth()

        self.jamming_intensity = 0.0
        self.comms_degradation = 0.0
