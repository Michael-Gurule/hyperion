"""Models module for HYPERION."""

from .detection import ThreatDetector, SensorFusion
from .gnn_communication import (
    AgentNodeEncoder,
    GNNCommunicationLayer,
    SwarmGNN,
    CoordinationHead,
    SwarmCoordinationNetwork,
)
from .adaptive_sensor_fusion import (
    AdaptiveKalmanFilter,
    BayesianSensorFusion,
    MultiHypothesisTracker,
    SensorMeasurement,
    SensorType,
)

__all__ = [
    # Detection models
    "ThreatDetector",
    "SensorFusion",
    # GNN communication
    "AgentNodeEncoder",
    "GNNCommunicationLayer",
    "SwarmGNN",
    "CoordinationHead",
    "SwarmCoordinationNetwork",
    # Adaptive sensor fusion
    "AdaptiveKalmanFilter",
    "BayesianSensorFusion",
    "MultiHypothesisTracker",
    "SensorMeasurement",
    "SensorType",
]
