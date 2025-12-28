"""
Threat detection and sensor fusion models.

Implements multi-modal sensor fusion for hypersonic threat detection,
building on techniques from SENTINEL (TDOA/FDOA) and CONSTELLATION (telemetry analysis).
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass


@dataclass
class SensorReading:
    """Container for multi-modal sensor data."""

    rf_signal: np.ndarray  # RF fingerprint features
    thermal_signature: np.ndarray  # IR/thermal features
    position_estimate: Optional[np.ndarray] = None  # TDOA/FDOA position
    confidence: float = 0.0
    timestamp: float = 0.0


class SensorFusion:
    """
    Multi-sensor fusion using Kalman filtering.
    Combines RF, thermal, and position estimates for robust tracking.
    """

    def __init__(
        self,
        state_dim: int = 4,  # [x, y, vx, vy]
        process_noise: float = 0.1,
        measurement_noise: float = 1.0,
    ):
        """
        Initialize Kalman filter for sensor fusion.

        Args:
            state_dim: Dimensionality of state vector
            process_noise: Process noise covariance
            measurement_noise: Measurement noise covariance
        """
        self.state_dim = state_dim

        # State: [x, y, vx, vy]
        self.state = np.zeros(state_dim)

        # State covariance
        self.P = np.eye(state_dim) * 100.0

        # Process noise
        self.Q = np.eye(state_dim) * process_noise

        # Measurement noise
        self.R = np.eye(2) * measurement_noise  # Only measure position

        # State transition matrix (constant velocity model)
        self.F = np.eye(state_dim)

        # Measurement matrix (observe position only)
        self.H = np.zeros((2, state_dim))
        self.H[0, 0] = 1.0  # Observe x
        self.H[1, 1] = 1.0  # Observe y

        # Initialized flag
        self.initialized = False

    def predict(self, dt: float):
        """
        Predict next state.

        Args:
            dt: Time step
        """
        # Update state transition for velocity
        self.F[0, 2] = dt
        self.F[1, 3] = dt

        # Predict state
        self.state = self.F @ self.state

        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(
        self, measurement: np.ndarray, measurement_noise: Optional[float] = None
    ):
        """
        Update state with new measurement.

        Args:
            measurement: Position measurement [x, y]
            measurement_noise: Optional custom measurement noise
        """
        if not self.initialized:
            # Initialize state with first measurement
            self.state[0:2] = measurement
            self.initialized = True
            return

        # Update measurement noise if provided
        R = self.R.copy()
        if measurement_noise is not None:
            R = np.eye(2) * measurement_noise

        # Innovation
        y = measurement - self.H @ self.state

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.state = self.state + K @ y

        # Update covariance
        I = np.eye(self.state_dim)
        self.P = (I - K @ self.H) @ self.P

    def fuse_measurements(
        self,
        rf_position: Optional[np.ndarray],
        thermal_position: Optional[np.ndarray],
        rf_confidence: float = 0.5,
        thermal_confidence: float = 0.5,
    ) -> Tuple[np.ndarray, float]:
        """
        Fuse multiple sensor measurements with confidence weighting.

        Args:
            rf_position: Position estimate from RF sensors
            thermal_position: Position estimate from thermal sensors
            rf_confidence: Confidence in RF measurement
            thermal_confidence: Confidence in thermal measurement

        Returns:
            Fused position estimate and overall confidence
        """
        measurements = []
        confidences = []

        if rf_position is not None:
            measurements.append(rf_position)
            confidences.append(rf_confidence)

        if thermal_position is not None:
            measurements.append(thermal_position)
            confidences.append(thermal_confidence)

        if len(measurements) == 0:
            return self.state[0:2], 0.0

        # Weighted average
        total_confidence = sum(confidences)
        weights = np.array(confidences) / (total_confidence + 1e-6)

        fused_position = np.zeros(2)
        for measurement, weight in zip(measurements, weights):
            fused_position += weight * measurement

        # Update Kalman filter
        measurement_noise = 1.0 / (total_confidence + 1e-6)
        self.update(fused_position, measurement_noise)

        return self.state[0:2], total_confidence / len(measurements)

    def get_state(self) -> np.ndarray:
        """Get current state estimate."""
        return self.state.copy()

    def get_position(self) -> np.ndarray:
        """Get current position estimate."""
        return self.state[0:2]

    def get_velocity(self) -> np.ndarray:
        """Get current velocity estimate."""
        return self.state[2:4]

    def reset(self):
        """Reset filter to initial state."""
        self.state = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim) * 100.0
        self.initialized = False


class ThreatDetector(nn.Module):
    """
    Neural network for threat detection and classification.
    Multi-modal input processing for RF and thermal signatures.
    """

    def __init__(
        self,
        rf_feature_dim: int = 128,
        thermal_feature_dim: int = 128,
        hidden_dim: int = 256,
        num_threat_classes: int = 3,  # hypersonic, supersonic, subsonic
    ):
        """
        Initialize threat detector network.

        Args:
            rf_feature_dim: Dimension of RF features
            thermal_feature_dim: Dimension of thermal features
            hidden_dim: Hidden layer dimension
            num_threat_classes: Number of threat classes
        """
        super().__init__()

        # RF signal processing branch
        self.rf_encoder = nn.Sequential(
            nn.Linear(rf_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Thermal signature processing branch
        self.thermal_encoder = nn.Sequential(
            nn.Linear(thermal_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Classification head
        self.classifier = nn.Linear(hidden_dim // 2, num_threat_classes)

        # Detection confidence head
        self.confidence = nn.Sequential(nn.Linear(hidden_dim // 2, 1), nn.Sigmoid())

    def forward(
        self, rf_features: torch.Tensor, thermal_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through detector.

        Args:
            rf_features: RF signal features [batch, rf_feature_dim]
            thermal_features: Thermal signature features [batch, thermal_feature_dim]

        Returns:
            Classification logits [batch, num_classes]
            Detection confidence [batch, 1]
        """
        # Encode each modality
        rf_encoded = self.rf_encoder(rf_features)
        thermal_encoded = self.thermal_encoder(thermal_features)

        # Concatenate and fuse
        fused = torch.cat([rf_encoded, thermal_encoded], dim=1)
        fused = self.fusion(fused)

        # Classification and confidence
        class_logits = self.classifier(fused)
        confidence = self.confidence(fused)

        return class_logits, confidence

    def detect(
        self,
        rf_features: np.ndarray,
        thermal_features: np.ndarray,
        threshold: float = 0.5,
    ) -> Tuple[bool, int, float]:
        """
        Detect and classify threat.

        Args:
            rf_features: RF signal features
            thermal_features: Thermal signature features
            threshold: Detection confidence threshold

        Returns:
            detected: Whether threat detected
            threat_class: Predicted threat class
            confidence: Detection confidence
        """
        self.eval()

        with torch.no_grad():
            # Convert to tensors
            rf_tensor = torch.FloatTensor(rf_features).unsqueeze(0)
            thermal_tensor = torch.FloatTensor(thermal_features).unsqueeze(0)

            # Forward pass
            class_logits, conf = self.forward(rf_tensor, thermal_tensor)

            # Get predictions
            threat_class = torch.argmax(class_logits, dim=1).item()
            confidence = conf.item()

            detected = confidence > threshold

            return detected, threat_class, confidence


class MultiTargetTracker:
    """
    Track multiple targets using multiple hypothesis tracking.
    Handles track initialization, update, and deletion.
    """

    def __init__(
        self,
        max_tracks: int = 10,
        association_threshold: float = 500.0,  # meters
        track_confirmation_threshold: int = 3,
        track_deletion_threshold: int = 5,
    ):
        """
        Initialize multi-target tracker.

        Args:
            max_tracks: Maximum number of simultaneous tracks
            association_threshold: Maximum distance for data association
            track_confirmation_threshold: Hits needed to confirm track
            track_deletion_threshold: Misses before deleting track
        """
        self.max_tracks = max_tracks
        self.association_threshold = association_threshold
        self.track_confirmation_threshold = track_confirmation_threshold
        self.track_deletion_threshold = track_deletion_threshold

        # Active tracks
        self.tracks: List[Dict] = []
        self.next_track_id = 0

    def update(self, measurements: List[np.ndarray], dt: float) -> List[Dict]:
        """
        Update tracks with new measurements.

        Args:
            measurements: List of position measurements
            dt: Time since last update

        Returns:
            List of confirmed tracks
        """
        # Predict all tracks
        for track in self.tracks:
            track["filter"].predict(dt)
            track["time_since_update"] += dt

        # Associate measurements to tracks
        unassociated_measurements = measurements.copy()

        for track in self.tracks:
            if len(unassociated_measurements) == 0:
                break

            # Find nearest measurement
            predicted_pos = track["filter"].get_position()
            distances = [
                np.linalg.norm(m - predicted_pos) for m in unassociated_measurements
            ]

            min_dist_idx = np.argmin(distances)
            min_dist = distances[min_dist_idx]

            if min_dist < self.association_threshold:
                # Associate measurement to track
                measurement = unassociated_measurements.pop(min_dist_idx)
                track["filter"].update(measurement)
                track["hits"] += 1
                track["time_since_update"] = 0.0

                # Confirm track if enough hits
                if track["hits"] >= self.track_confirmation_threshold:
                    track["confirmed"] = True

        # Create new tracks for unassociated measurements
        for measurement in unassociated_measurements:
            if len(self.tracks) < self.max_tracks:
                new_track = {
                    "id": self.next_track_id,
                    "filter": SensorFusion(),
                    "hits": 1,
                    "time_since_update": 0.0,
                    "confirmed": False,
                }
                new_track["filter"].update(measurement)
                self.tracks.append(new_track)
                self.next_track_id += 1

        # Delete old tracks
        self.tracks = [
            track
            for track in self.tracks
            if track["time_since_update"] < self.track_deletion_threshold
        ]

        # Return confirmed tracks
        return [track for track in self.tracks if track["confirmed"]]

    def get_tracks(self) -> List[Dict]:
        """Get all confirmed tracks."""
        return [track for track in self.tracks if track["confirmed"]]

    def reset(self):
        """Reset tracker."""
        self.tracks = []
        self.next_track_id = 0
