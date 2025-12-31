# Core Components

This page provides detailed documentation for each major component in HYPERION.

---

## 1. HypersonicSwarmEnv

**File**: `src/env/hypersonic_swarm_env.py`

The core simulation environment implementing the PettingZoo ParallelEnv interface.

### Overview

```python
from src.env.hypersonic_swarm_env import HypersonicSwarmEnv

env = HypersonicSwarmEnv(
    num_agents=5,
    target_speed=1700.0,  # Mach 5
    detection_range=2000.0
)
```

### State Space

Each agent observes a 25-dimensional vector:

| Dimensions | Content | Range |
|------------|---------|-------|
| 0-1 | Own position (x, y) | Normalized to arena |
| 2-3 | Own velocity (vx, vy) | Normalized to max speed |
| 4 | Fuel level | [0, 1] |
| 5-6 | Target relative position | If detected |
| 7-8 | Target velocity estimate | If detected |
| 9-24 | Nearest neighbor states (up to 3) | Position + velocity each |

### Action Space

Continuous 2D action space:

| Dimension | Meaning | Range |
|-----------|---------|-------|
| 0 | Thrust magnitude | [0, 1] |
| 1 | Heading change | [-1, 1] (angular acceleration) |

### Physics Constants

```python
MAX_AGENT_SPEED = 300.0      # m/s
MAX_TARGET_SPEED = 1700.0    # m/s (Mach 5)
MAX_ACCELERATION = 50.0      # m/s²
MAX_TURN_RATE = np.pi / 4    # rad/s
DETECTION_RANGE = 2000.0     # meters
INTERCEPT_RANGE = 50.0       # meters
COMMUNICATION_RANGE = 1500.0 # meters
DT = 0.1                     # seconds
```

### Reward Structure

```python
# Terminal rewards
INTERCEPT_REWARD = +100.0    # Any agent intercepts target
ESCAPE_PENALTY = -100.0      # Target reaches boundary

# Shaping rewards (per step)
distance_penalty = -0.01 * min_distance_to_target
fuel_penalty = -0.1 * fuel_consumed
formation_bonus = +0.1 * formation_quality
```

### Episode Termination

An episode ends when:
1. **Success**: Any agent gets within `INTERCEPT_RANGE` of target
2. **Failure**: Target crosses arena boundary
3. **Timeout**: Maximum steps reached (1000 default)

---

## 2. Physics Models

**File**: `src/env/physics_models.py`

Realistic atmospheric and trajectory physics.

### AtmosphereModel

Implements the International Standard Atmosphere (ISA):

```python
from src.env.physics_models import AtmosphereModel

# Get conditions at 10km altitude
conditions = AtmosphereModel.get_conditions(10000)
print(f"Temperature: {conditions.temperature} K")
print(f"Pressure: {conditions.pressure} Pa")
print(f"Density: {conditions.density} kg/m³")
print(f"Speed of sound: {conditions.speed_of_sound} m/s")

# Convert Mach to velocity
velocity = AtmosphereModel.mach_to_velocity(5.0, altitude=10000)
```

**Key Equations**:
- Temperature: `T = T₀ + L·h` (lapse rate model)
- Pressure: Barometric formula
- Density: Ideal gas law
- Speed of sound: `a = √(γRT)`

### HypersonicTrajectory

Trajectory simulation with atmospheric drag:

```python
from src.env.physics_models import HypersonicTrajectory
import numpy as np

trajectory = HypersonicTrajectory(
    initial_position=np.array([0.0, 0.0]),
    initial_velocity=np.array([2000.0, -500.0]),
    mass=1000.0,
    drag_coefficient=0.3,
    reference_area=0.5
)

# Simulate for 10 seconds
for _ in range(100):
    trajectory.update(dt=0.1, altitude=5000)

position, velocity = trajectory.get_state()
```

### SensorModel

Sensor physics with noise and detection probability:

```python
from src.env.physics_models import SensorModel

# Detection probability decreases with range
prob = SensorModel.detection_probability(
    distance=15000,      # 15 km
    max_range=30000,     # 30 km max range
    base_probability=0.95
)

# Add measurement noise
noisy_pos = SensorModel.add_measurement_noise(
    measurement=true_position,
    noise_std=50.0  # meters
)
```

### EvasiveManeuvers

Target evasion behaviors:

```python
from src.env.physics_models import EvasiveManeuvers

# Sinusoidal weaving
evasive_vel = EvasiveManeuvers.sinusoidal_weave(
    base_velocity, time, amplitude=100.0, frequency=0.2
)

# Random jinking
jink_vel = EvasiveManeuvers.random_jink(
    base_velocity, probability=0.05, magnitude=50.0
)
```

---

## 3. Projectile System

**File**: `src/env/projectile_system.py`

Guided interceptor missiles with proportional navigation.

### Proportional Navigation

The classic aerospace guidance law:

```
a_cmd = N × V_c × (dλ/dt)
```

Where:
- `N` = Navigation constant (3.0 typical)
- `V_c` = Closing velocity
- `dλ/dt` = Line-of-sight rate

```python
from src.env.projectile_system import ProportionalNavigationGuidance

guidance = ProportionalNavigationGuidance(navigation_constant=3.0)

# Compute guidance command
acceleration = guidance.compute_acceleration_command(
    projectile_pos, projectile_vel,
    target_pos, target_vel
)
```

### ProjectileManager

Lifecycle management for missiles:

```python
from src.env.projectile_system import ProjectileManager

manager = ProjectileManager(
    projectile_speed=600.0,  # m/s
    max_lifetime=5.0,        # seconds
    hit_radius=20.0,         # meters
    cooldown=2.0             # seconds between launches
)

# Launch projectile
manager.launch(agent_id="uav_0", position, velocity, target_pos)

# Update all projectiles
hits = manager.update(dt=0.1, target_pos, target_vel)

# Check for hits
if hits:
    print(f"Target hit by agent: {hits[0]}")
```

### Why Proportional Navigation?

PN guidance was chosen because:
1. **Proven**: Used in real missile systems for decades
2. **Robust**: Works against maneuvering targets
3. **Simple**: Requires only LOS measurements
4. **Optimal**: Minimizes miss distance for constant-velocity targets

---

## 4. Detection Module

**File**: `src/models/detection.py`

Multi-modal threat detection and tracking.

### SensorFusion (Kalman Filter)

```python
from src.models.detection import SensorFusion

fusion = SensorFusion(process_noise=0.1, measurement_noise=50.0)

# Initialize with first detection
fusion.initialize(initial_position)

# Update with new measurements
for measurement in measurements:
    fusion.update(measurement)
    estimated_state = fusion.get_state()  # [x, y, vx, vy]
    confidence = fusion.get_confidence()
```

### ThreatDetector (Neural Network)

Multi-branch architecture for sensor fusion:

```
RF Signal ──────┐
(128 dims)      │
      ↓         │
  FC(128→128)   │
      ↓         │
  FC(128→64)  ──┼──→ Concat ──→ FC(128→64) ──→ Classification
      ↓         │                              (3 classes)
Thermal ────────┘
(128 dims)
```

```python
from src.models.detection import ThreatDetector

detector = ThreatDetector()

# Classify threat
classification, confidence = detector.classify(
    rf_features, thermal_features
)
# classification: 0=hypersonic, 1=supersonic, 2=subsonic
```

### MultiTargetTracker

Track management with data association:

```python
from src.models.detection import MultiTargetTracker

tracker = MultiTargetTracker(
    max_tracks=10,
    confirmation_threshold=3,
    deletion_threshold=5
)

# Update with detections
tracker.update(detections)

# Get confirmed tracks
confirmed_tracks = tracker.get_confirmed_tracks()
```

---

## 5. Hierarchical Policy

**File**: `src/models/hierarchical_policy.py`

Role-based coordination for specialized agent behaviors.

### Agent Roles

| Role | ID | Behavior |
|------|----|----------|
| SCOUT | 0 | Maximize detection coverage, stay distant |
| TRACKER | 1 | Maintain line-of-sight, medium range |
| INTERCEPTOR | 2 | Close approach, prepare firing |
| SUPPORT | 3 | Backup and gap filling |

### Architecture

```
                    ┌─────────────────┐
                    │  GNN Embeddings │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │  Role    │  │  Role    │  │  Role    │
        │ Assigner │  │ Policy 0 │  │ Policy 1 │  ...
        │ (Manager)│  │ (Scout)  │  │(Tracker) │
        └────┬─────┘  └────┬─────┘  └────┬─────┘
             │              │              │
             ▼              └──────┬───────┘
        Role Selection             ▼
                             Role-Specific
                               Actions
```

```python
from src.models.hierarchical_policy import RoleBasedOptionCritic

policy = RoleBasedOptionCritic(
    obs_dim=25,
    action_dim=2,
    num_roles=4,
    gnn_hidden_dim=64
)

# Get action with role assignment
action, role = policy.get_action(observation, agent_embeddings)
```

---

## 6. Curriculum Learning

**File**: `src/training/curriculum.py`

Progressive difficulty scaling for stable training.

### Stages

| Stage | Target Speed | Evasion | Agents |
|-------|--------------|---------|--------|
| Basic | 500 m/s (subsonic) | None | 3 |
| Intermediate | 1000 m/s (supersonic) | Basic weave | 5 |
| Advanced | 1700 m/s (hypersonic) | Full evasion | 5 |

### Advancement Logic

```python
from src.training.curriculum import CurriculumScheduler

scheduler = CurriculumScheduler(
    success_threshold=0.7,    # 70% interception to advance
    min_episodes=100,         # Minimum episodes per stage
    window_size=50            # Rolling average window
)

# After each episode
scheduler.record_episode(success=True)

if scheduler.should_advance():
    scheduler.advance_stage()
    new_config = scheduler.get_current_config()
```

---

## 7. Evaluation Metrics

**File**: `src/evaluation/metrics.py`

Comprehensive performance tracking.

### Available Metrics

| Metric | Description |
|--------|-------------|
| `interception_rate` | Percentage of successful interceptions |
| `mean_episode_length` | Average steps per episode |
| `mean_reward` | Average cumulative reward |
| `fuel_efficiency` | Fuel remaining at interception |
| `min_distance` | Closest approach to target |
| `time_to_intercept` | Steps until interception |

```python
from src.evaluation.metrics import EvaluationMetrics

metrics = EvaluationMetrics()

# Record episodes
for episode in episodes:
    metrics.add_episode(episode_metrics)

# Get summary
summary = metrics.get_summary()
print(f"Interception Rate: {summary['interception_rate']:.1%}")

# Export to JSON
metrics.export("results.json")
```

---

## 8. Dashboard

**File**: `src/dashboard/app.py`

Interactive Streamlit web interface.

### Features

1. **Training Progress**: Loss curves, reward trends
2. **Live Simulation**: Real-time swarm visualization
3. **Batch Evaluation**: Run multiple episodes
4. **Curriculum Metrics**: Stage progression tracking
5. **Role Distribution**: Agent specialization analysis

### Usage

```bash
streamlit run src/dashboard/app.py
```

Access at `http://localhost:8501`

---

## 9. ARSHI: Autonomous Resilient Sensing & Hive Intelligence

**Files**: `src/models/arshi.py`, `src/models/opportunistic_sensors.py`, `src/models/belief_system.py`, `src/models/resilient_gnn.py`

Enables operation in contested electromagnetic environments where radar and GPS are degraded or denied.

### Overview

ARSHI provides four integrated layers:
1. **Opportunistic Sensors**: 6 unconventional sensing modalities
2. **Bayesian Belief System**: Probabilistic threat tracking with uncertainty
3. **Gossip Protocol**: Comms-resilient information sharing
4. **Resilient GNN**: Degradation-aware swarm coordination

### Opportunistic Sensor Suite

| Sensor | Signal Source | Jamming Immunity |
|--------|---------------|------------------|
| Plasma Emission | Hypersonic plasma sheath RF | High |
| Acoustic Array | Sonic boom triangulation | Complete |
| Thermal Wake | Atmospheric heating signatures | High |
| Magnetic Anomaly | Metallic mass perturbations | Complete |
| Passive Coherent Location | Ambient RF (cell/FM) scatter | Medium |
| Swarm Proprioception | Neighbor behavioral inference | Complete |

```python
from src.models.opportunistic_sensors import OpportunisticSensorSuite

sensors = OpportunisticSensorSuite(
    enable_plasma=True,
    enable_acoustic=True,
    enable_thermal=True,
    enable_magnetic=True,
    enable_pcl=True,
    enable_proprioception=True
)

# Get all available measurements
measurements = sensors.sense(
    target_position=np.array([1000, 2000]),
    target_velocity=np.array([1700, 0]),
    agent_position=np.array([0, 0]),
    neighbor_data=neighbor_info
)
```

### Belief System with Gossip Protocol

```python
from src.models.belief_system import SwarmBeliefManager

manager = SwarmBeliefManager(num_agents=5, state_dim=4)

# Update agent belief from measurements
manager.update_agent(agent_id=0, measurement, covariance)

# Generate gossip messages (epidemic sharing)
messages = manager.generate_gossip_messages(gossip_rate=0.3)

# Receive and fuse (uses covariance intersection)
for msg in messages:
    manager.receive_gossip(msg)

# Get fused estimate
estimate, confidence = manager.get_agent_estimate(agent_id=0)
```

Key property: Converges even with **90%+ packet loss**.

### Operating Modes

```
FULL → DEGRADED → MINIMAL → PROPRIOCEPTIVE → ISOLATED
```

Automatic mode switching based on sensor/communication availability.

### Swarm Proprioception (Novel Contribution)

Infers threat location by observing neighbor behavior changes:

```python
from src.models.opportunistic_sensors import SwarmProprioception

proprio = SwarmProprioception(observation_window=20, min_neighbors=2)

# Detect coordinated behavior changes
measurement = proprio.infer_from_neighbors(
    neighbor_positions=[...],
    neighbor_velocities=[...],
    neighbor_history=[...]
)
```

This enables threat detection even when all direct sensors are denied.

### Full Integration

```python
from src.models.arshi import ARSHISwarm, ARSHIConfig

config = ARSHIConfig(num_agents=5, gossip_rate=0.3)
swarm = ARSHISwarm(config)

# Process environment step
observations = swarm.process_step(
    agent_positions=[...],
    agent_velocities=[...],
    target_position=target_pos,
    target_velocity=target_vel,
    comm_adjacency=adj_matrix
)

# Each observation contains:
# - target_estimate, target_uncertainty
# - sensor_readings, operating_mode
# - is_isolated flag
```

See [ARSHI System](ARSHI-System) for comprehensive documentation
