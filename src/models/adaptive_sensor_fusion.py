"""
Adaptive Sensor Fusion with Uncertainty Quantification.

Implements advanced Kalman filtering with:
- Adaptive noise estimation from innovation sequences
- Bayesian uncertainty quantification
- Multi-hypothesis tracking for ambiguous scenarios
- Sensor reliability estimation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats
from collections import deque


class SensorType(Enum):
    """Types of sensors in the system."""
    RF = "rf"
    THERMAL = "thermal"
    RADAR = "radar"
    LIDAR = "lidar"
    ACOUSTIC = "acoustic"


@dataclass
class SensorMeasurement:
    """Container for sensor measurements with metadata."""
    position: np.ndarray
    sensor_type: SensorType
    timestamp: float
    raw_confidence: float = 1.0
    sensor_id: Optional[str] = None

    # Optional velocity measurement
    velocity: Optional[np.ndarray] = None

    # Measurement covariance (if known)
    covariance: Optional[np.ndarray] = None


@dataclass
class TrackState:
    """State of a tracked target."""
    state: np.ndarray  # [x, y, vx, vy, ax, ay] for 6D or [x, y, vx, vy] for 4D
    covariance: np.ndarray
    track_id: int
    age: int = 0
    hits: int = 0
    misses: int = 0
    confirmed: bool = False

    # Uncertainty quantification
    position_uncertainty: float = 0.0
    velocity_uncertainty: float = 0.0

    # Track quality metrics
    innovation_history: List[float] = field(default_factory=list)
    nees_history: List[float] = field(default_factory=list)


class AdaptiveKalmanFilter:
    """
    Adaptive Kalman Filter with automatic noise estimation.

    Features:
    - Innovation-based adaptive Q/R estimation
    - Fading memory for non-stationary systems
    - Outlier rejection via Mahalanobis distance
    """

    def __init__(
        self,
        state_dim: int = 6,  # [x, y, vx, vy, ax, ay]
        measurement_dim: int = 2,  # [x, y]
        process_noise: float = 1.0,
        measurement_noise: float = 10.0,
        adaptation_rate: float = 0.1,
        fading_factor: float = 1.0,
        innovation_window: int = 20,
    ):
        """
        Initialize adaptive Kalman filter.

        Args:
            state_dim: State vector dimension
            measurement_dim: Measurement dimension
            process_noise: Initial process noise
            measurement_noise: Initial measurement noise
            adaptation_rate: Rate of noise adaptation
            fading_factor: Fading memory factor (>= 1.0)
            innovation_window: Window for innovation statistics
        """
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.adaptation_rate = adaptation_rate
        self.fading_factor = fading_factor
        self.innovation_window = innovation_window

        # State
        self.state = np.zeros(state_dim)
        self.P = np.eye(state_dim) * 100.0

        # Process noise (adaptive)
        self.Q_base = np.eye(state_dim) * process_noise
        self.Q = self.Q_base.copy()

        # Measurement noise (adaptive)
        self.R_base = np.eye(measurement_dim) * measurement_noise
        self.R = self.R_base.copy()

        # State transition matrix (constant acceleration model)
        self.F = self._build_transition_matrix(0.1)

        # Measurement matrix
        self.H = np.zeros((measurement_dim, state_dim))
        self.H[0, 0] = 1.0  # x
        self.H[1, 1] = 1.0  # y

        # Innovation history for adaptation
        self.innovation_history = deque(maxlen=innovation_window)
        self.innovation_covariance = np.eye(measurement_dim)

        # State tracking
        self.initialized = False
        self.step_count = 0

    def _build_transition_matrix(self, dt: float) -> np.ndarray:
        """Build state transition matrix for given timestep."""
        F = np.eye(self.state_dim)

        if self.state_dim >= 4:
            # Velocity integration
            F[0, 2] = dt  # x += vx * dt
            F[1, 3] = dt  # y += vy * dt

        if self.state_dim >= 6:
            # Acceleration integration
            F[0, 4] = 0.5 * dt ** 2  # x += 0.5 * ax * dt^2
            F[1, 5] = 0.5 * dt ** 2  # y += 0.5 * ay * dt^2
            F[2, 4] = dt  # vx += ax * dt
            F[3, 5] = dt  # vy += ay * dt

        return F

    def predict(self, dt: float):
        """Predict next state with process noise adaptation."""
        # Update transition matrix
        self.F = self._build_transition_matrix(dt)

        # Predict state
        self.state = self.F @ self.state

        # Predict covariance with fading memory
        self.P = self.fading_factor * (self.F @ self.P @ self.F.T) + self.Q

        self.step_count += 1

    def update(
        self,
        measurement: np.ndarray,
        measurement_covariance: Optional[np.ndarray] = None,
        outlier_threshold: float = 5.0,
    ) -> Tuple[bool, float]:
        """
        Update state with measurement, rejecting outliers.

        Args:
            measurement: Position measurement [x, y]
            measurement_covariance: Per-measurement covariance (optional)
            outlier_threshold: Mahalanobis distance threshold

        Returns:
            accepted: Whether measurement was accepted
            mahalanobis_distance: Distance metric
        """
        if not self.initialized:
            self._initialize(measurement)
            return True, 0.0

        # Use provided or adaptive R
        R = measurement_covariance if measurement_covariance is not None else self.R

        # Innovation
        innovation = measurement - self.H @ self.state

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + R

        # Mahalanobis distance for outlier detection
        try:
            S_inv = np.linalg.inv(S)
            mahalanobis = np.sqrt(innovation @ S_inv @ innovation)
        except np.linalg.LinAlgError:
            mahalanobis = np.linalg.norm(innovation)

        # Outlier rejection
        if mahalanobis > outlier_threshold:
            return False, mahalanobis

        # Kalman gain
        K = self.P @ self.H.T @ S_inv

        # Update state
        self.state = self.state + K @ innovation

        # Update covariance (Joseph form for numerical stability)
        I_KH = np.eye(self.state_dim) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        # Store innovation for adaptation
        self.innovation_history.append(innovation)
        self._adapt_noise()

        return True, mahalanobis

    def _initialize(self, measurement: np.ndarray):
        """Initialize filter with first measurement."""
        self.state[:2] = measurement
        self.initialized = True

    def _adapt_noise(self):
        """Adapt process and measurement noise from innovations."""
        if len(self.innovation_history) < self.innovation_window // 2:
            return

        innovations = np.array(list(self.innovation_history))

        # Estimate innovation covariance
        sample_cov = np.cov(innovations.T)
        if sample_cov.ndim == 0:
            sample_cov = np.array([[sample_cov]])

        # Expected innovation covariance
        expected_cov = self.H @ self.P @ self.H.T + self.R

        # Adaptation ratio
        try:
            ratio = np.trace(sample_cov) / np.trace(expected_cov)
        except Exception:
            ratio = 1.0

        # Adapt R (measurement noise)
        if ratio > 1.2:
            # Innovations larger than expected -> increase R
            self.R = (1 - self.adaptation_rate) * self.R + self.adaptation_rate * sample_cov
        elif ratio < 0.8:
            # Innovations smaller than expected -> decrease R (slowly)
            self.R = (1 - self.adaptation_rate * 0.5) * self.R + self.adaptation_rate * 0.5 * self.R_base

        # Adapt Q (process noise) based on state covariance growth
        if np.trace(self.P) > np.trace(self.Q) * 100:
            # Covariance growing too fast -> increase Q
            self.Q *= 1.0 + self.adaptation_rate

    def get_position(self) -> np.ndarray:
        """Get current position estimate."""
        return self.state[:2].copy()

    def get_velocity(self) -> np.ndarray:
        """Get current velocity estimate."""
        if self.state_dim >= 4:
            return self.state[2:4].copy()
        return np.zeros(2)

    def get_acceleration(self) -> np.ndarray:
        """Get current acceleration estimate."""
        if self.state_dim >= 6:
            return self.state[4:6].copy()
        return np.zeros(2)

    def get_position_uncertainty(self) -> float:
        """Get position uncertainty (sqrt of trace of position covariance)."""
        return np.sqrt(self.P[0, 0] + self.P[1, 1])

    def get_velocity_uncertainty(self) -> float:
        """Get velocity uncertainty."""
        if self.state_dim >= 4:
            return np.sqrt(self.P[2, 2] + self.P[3, 3])
        return 0.0

    def compute_nees(self, true_state: np.ndarray) -> float:
        """
        Compute Normalized Estimation Error Squared (NEES).

        Args:
            true_state: Ground truth state

        Returns:
            NEES value (should be close to state_dim if consistent)
        """
        error = self.state - true_state[:self.state_dim]
        try:
            P_inv = np.linalg.inv(self.P)
            nees = error @ P_inv @ error
        except np.linalg.LinAlgError:
            nees = np.sum(error ** 2)
        return nees

    def reset(self):
        """Reset filter to initial state."""
        self.state = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim) * 100.0
        self.Q = self.Q_base.copy()
        self.R = self.R_base.copy()
        self.innovation_history.clear()
        self.initialized = False
        self.step_count = 0


class BayesianSensorFusion:
    """
    Bayesian sensor fusion with uncertainty quantification.

    Fuses measurements from multiple sensor types with learned reliability.
    """

    def __init__(
        self,
        sensor_types: List[SensorType] = None,
        prior_reliability: float = 0.8,
        reliability_adaptation_rate: float = 0.05,
    ):
        """
        Initialize Bayesian sensor fusion.

        Args:
            sensor_types: List of sensor types to track
            prior_reliability: Initial reliability estimate
            reliability_adaptation_rate: Rate of reliability learning
        """
        if sensor_types is None:
            sensor_types = [SensorType.RF, SensorType.THERMAL]

        self.sensor_types = sensor_types
        self.adaptation_rate = reliability_adaptation_rate

        # Per-sensor reliability estimates (Beta distribution parameters)
        self.sensor_reliability = {
            st: {"alpha": prior_reliability * 10, "beta": (1 - prior_reliability) * 10}
            for st in sensor_types
        }

        # Per-sensor noise estimates
        self.sensor_noise = {
            st: 10.0  # Initial measurement noise estimate
            for st in sensor_types
        }

        # Adaptive Kalman filter
        self.kalman = AdaptiveKalmanFilter(
            state_dim=6,
            measurement_dim=2,
            process_noise=1.0,
            measurement_noise=10.0,
        )

        # Fusion statistics
        self.fusion_history = []

    def get_sensor_reliability(self, sensor_type: SensorType) -> float:
        """Get reliability estimate for sensor type."""
        params = self.sensor_reliability.get(
            sensor_type,
            {"alpha": 5.0, "beta": 5.0}
        )
        # Mean of Beta distribution
        return params["alpha"] / (params["alpha"] + params["beta"])

    def get_sensor_uncertainty(self, sensor_type: SensorType) -> float:
        """Get uncertainty in reliability estimate."""
        params = self.sensor_reliability.get(
            sensor_type,
            {"alpha": 5.0, "beta": 5.0}
        )
        # Variance of Beta distribution
        a, b = params["alpha"], params["beta"]
        variance = (a * b) / ((a + b) ** 2 * (a + b + 1))
        return np.sqrt(variance)

    def update_sensor_reliability(
        self,
        sensor_type: SensorType,
        measurement_accepted: bool,
    ):
        """Update sensor reliability based on measurement acceptance."""
        if sensor_type not in self.sensor_reliability:
            self.sensor_reliability[sensor_type] = {"alpha": 5.0, "beta": 5.0}

        params = self.sensor_reliability[sensor_type]

        if measurement_accepted:
            params["alpha"] += self.adaptation_rate
        else:
            params["beta"] += self.adaptation_rate

        # Prevent extreme values
        total = params["alpha"] + params["beta"]
        if total > 100:
            # Normalize to prevent overflow
            params["alpha"] = params["alpha"] / total * 100
            params["beta"] = params["beta"] / total * 100

    def fuse_measurements(
        self,
        measurements: List[SensorMeasurement],
        dt: float,
    ) -> Tuple[np.ndarray, float, Dict[str, float]]:
        """
        Fuse multiple sensor measurements.

        Args:
            measurements: List of sensor measurements
            dt: Time step since last fusion

        Returns:
            fused_position: Fused position estimate
            confidence: Overall fusion confidence
            sensor_contributions: Per-sensor contribution weights
        """
        # Predict
        self.kalman.predict(dt)

        if len(measurements) == 0:
            return (
                self.kalman.get_position(),
                1.0 / (1.0 + self.kalman.get_position_uncertainty()),
                {},
            )

        # Compute weights for each measurement
        weights = []
        for meas in measurements:
            reliability = self.get_sensor_reliability(meas.sensor_type)
            raw_confidence = meas.raw_confidence

            # Compute measurement noise based on sensor type
            noise = self.sensor_noise.get(meas.sensor_type, 10.0)

            # Weight = reliability * confidence / noise
            weight = reliability * raw_confidence / (noise + 1e-6)
            weights.append(weight)

        # Normalize weights
        total_weight = sum(weights) + 1e-6
        normalized_weights = [w / total_weight for w in weights]

        # Fused measurement (weighted average)
        fused_position = np.zeros(2)
        fused_covariance = np.zeros((2, 2))

        sensor_contributions = {}

        for meas, weight in zip(measurements, normalized_weights):
            fused_position += weight * meas.position

            # Accumulate covariance
            if meas.covariance is not None:
                fused_covariance += weight * meas.covariance
            else:
                noise = self.sensor_noise.get(meas.sensor_type, 10.0)
                fused_covariance += weight * np.eye(2) * noise

            # Track contributions
            sensor_name = meas.sensor_type.value
            sensor_contributions[sensor_name] = sensor_contributions.get(sensor_name, 0) + weight

        # Update Kalman filter with fused measurement
        accepted, mahalanobis = self.kalman.update(
            fused_position,
            measurement_covariance=fused_covariance,
        )

        # Update sensor reliabilities
        for meas in measurements:
            self.update_sensor_reliability(meas.sensor_type, accepted)

        # Compute overall confidence
        position_uncertainty = self.kalman.get_position_uncertainty()
        confidence = 1.0 / (1.0 + position_uncertainty / 100.0)

        # Store fusion result
        self.fusion_history.append({
            "position": self.kalman.get_position(),
            "confidence": confidence,
            "num_measurements": len(measurements),
            "accepted": accepted,
        })

        return self.kalman.get_position(), confidence, sensor_contributions

    def get_state(self) -> np.ndarray:
        """Get current full state estimate."""
        return self.kalman.state.copy()

    def get_position(self) -> np.ndarray:
        """Get current position estimate."""
        return self.kalman.get_position()

    def get_velocity(self) -> np.ndarray:
        """Get current velocity estimate."""
        return self.kalman.get_velocity()

    def get_position_uncertainty(self) -> float:
        """Get position uncertainty."""
        return self.kalman.get_position_uncertainty()

    def get_velocity_uncertainty(self) -> float:
        """Get velocity uncertainty."""
        return self.kalman.get_velocity_uncertainty()

    def get_confidence_interval(
        self,
        confidence_level: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get confidence interval for position estimate.

        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)

        Returns:
            lower: Lower bound of confidence interval
            upper: Upper bound of confidence interval
        """
        position = self.kalman.get_position()
        uncertainty = self.kalman.get_position_uncertainty()

        # Chi-squared quantile for 2D
        chi2_val = stats.chi2.ppf(confidence_level, df=2)
        radius = np.sqrt(chi2_val) * uncertainty

        lower = position - radius
        upper = position + radius

        return lower, upper

    def reset(self):
        """Reset fusion system."""
        self.kalman.reset()
        self.fusion_history.clear()


class MultiHypothesisTracker:
    """
    Multi-Hypothesis Tracking (MHT) for ambiguous scenarios.

    Maintains multiple track hypotheses when data association is uncertain.
    """

    def __init__(
        self,
        max_hypotheses: int = 5,
        max_tracks: int = 20,
        confirmation_threshold: int = 3,
        deletion_threshold: int = 5,
        gate_probability: float = 0.99,
    ):
        """
        Initialize MHT.

        Args:
            max_hypotheses: Maximum hypotheses per track
            max_tracks: Maximum simultaneous tracks
            confirmation_threshold: Hits to confirm track
            deletion_threshold: Misses to delete track
            gate_probability: Probability for gating
        """
        self.max_hypotheses = max_hypotheses
        self.max_tracks = max_tracks
        self.confirmation_threshold = confirmation_threshold
        self.deletion_threshold = deletion_threshold
        self.gate_probability = gate_probability

        # Gate size for chi-squared with 2 DOF
        self.gate_size = stats.chi2.ppf(gate_probability, df=2)

        self.tracks: List[TrackState] = []
        self.next_track_id = 0

        # Per-track Kalman filters
        self.filters: Dict[int, BayesianSensorFusion] = {}

    def update(
        self,
        measurements: List[SensorMeasurement],
        dt: float,
    ) -> List[TrackState]:
        """
        Update tracks with new measurements.

        Args:
            measurements: New sensor measurements
            dt: Time since last update

        Returns:
            List of confirmed tracks
        """
        # Predict all tracks
        for track in self.tracks:
            if track.track_id in self.filters:
                self.filters[track.track_id].kalman.predict(dt)
                track.age += 1

        # Compute gating and association
        associations = self._compute_associations(measurements)

        # Update associated tracks
        associated_measurements = set()

        for track_id, meas_indices in associations.items():
            if len(meas_indices) == 0:
                # Track miss
                track = self._get_track(track_id)
                if track:
                    track.misses += 1
                continue

            # Get measurements for this track
            track_measurements = [measurements[i] for i in meas_indices]
            associated_measurements.update(meas_indices)

            # Update filter
            if track_id in self.filters:
                fusion = self.filters[track_id]
                pos, conf, _ = fusion.fuse_measurements(track_measurements, dt)

                # Update track state
                track = self._get_track(track_id)
                if track:
                    track.state = fusion.get_state()
                    track.covariance = fusion.kalman.P.copy()
                    track.hits += 1
                    track.misses = 0
                    track.position_uncertainty = fusion.get_position_uncertainty()
                    track.velocity_uncertainty = fusion.get_velocity_uncertainty()

                    if track.hits >= self.confirmation_threshold:
                        track.confirmed = True

        # Create new tracks for unassociated measurements
        for i, meas in enumerate(measurements):
            if i not in associated_measurements:
                self._create_track(meas)

        # Delete old tracks
        self._prune_tracks()

        return [t for t in self.tracks if t.confirmed]

    def _compute_associations(
        self,
        measurements: List[SensorMeasurement],
    ) -> Dict[int, List[int]]:
        """
        Compute measurement-to-track associations using gating.
        """
        associations = {t.track_id: [] for t in self.tracks}

        for i, meas in enumerate(measurements):
            best_track = None
            best_distance = float("inf")

            for track in self.tracks:
                if track.track_id not in self.filters:
                    continue

                fusion = self.filters[track.track_id]
                predicted_pos = fusion.get_position()

                # Mahalanobis distance
                diff = meas.position - predicted_pos
                uncertainty = fusion.get_position_uncertainty()

                try:
                    P_pos = fusion.kalman.P[:2, :2]
                    P_inv = np.linalg.inv(P_pos)
                    mahal_sq = diff @ P_inv @ diff
                except np.linalg.LinAlgError:
                    mahal_sq = np.sum(diff ** 2) / (uncertainty ** 2 + 1e-6)

                # Gating
                if mahal_sq < self.gate_size:
                    if mahal_sq < best_distance:
                        best_distance = mahal_sq
                        best_track = track.track_id

            if best_track is not None:
                associations[best_track].append(i)

        return associations

    def _get_track(self, track_id: int) -> Optional[TrackState]:
        """Get track by ID."""
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None

    def _create_track(self, measurement: SensorMeasurement):
        """Create new track from measurement."""
        if len(self.tracks) >= self.max_tracks:
            return

        track_id = self.next_track_id
        self.next_track_id += 1

        # Initialize track state
        state = np.zeros(6)
        state[:2] = measurement.position
        covariance = np.eye(6) * 100.0

        track = TrackState(
            state=state,
            covariance=covariance,
            track_id=track_id,
            hits=1,
        )

        self.tracks.append(track)

        # Create filter
        fusion = BayesianSensorFusion()
        fusion.kalman.state[:2] = measurement.position
        fusion.kalman.initialized = True
        self.filters[track_id] = fusion

    def _prune_tracks(self):
        """Remove stale tracks."""
        to_remove = []

        for track in self.tracks:
            if track.misses >= self.deletion_threshold:
                to_remove.append(track.track_id)

        for track_id in to_remove:
            self.tracks = [t for t in self.tracks if t.track_id != track_id]
            if track_id in self.filters:
                del self.filters[track_id]

    def get_tracks(self) -> List[TrackState]:
        """Get all confirmed tracks."""
        return [t for t in self.tracks if t.confirmed]

    def get_all_tracks(self) -> List[TrackState]:
        """Get all tracks including tentative."""
        return self.tracks.copy()

    def reset(self):
        """Reset tracker."""
        self.tracks.clear()
        self.filters.clear()
        self.next_track_id = 0
