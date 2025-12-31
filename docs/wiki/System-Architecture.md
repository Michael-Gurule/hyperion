# System Architecture

HYPERION follows a layered architecture that separates concerns and enables modular development, testing, and deployment.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │    Dashboard    │  │   Evaluation    │  │   Notebooks     │  │
│  │   (Streamlit)   │  │    Metrics      │  │   (Jupyter)     │  │ 
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
└───────────┼─────────────────────┼─────────────────────┼─────────┘
            │                     │                     │
┌───────────┼─────────────────────┼─────────────────────┼─────────┐
│           │          TRAINING LAYER                   │         │
│  ┌────────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐  │
│  │   RLlib PPO     │  │   Curriculum    │  │    Intrinsic    │  │
│  │   (Ray)         │  │   Learning      │  │    Rewards      │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
└───────────┼─────────────────────┼─────────────────────┼─────────┘
            │                     │                     │
┌───────────┼─────────────────────┼─────────────────────┼─────────┐
│           │           MODEL LAYER                     │         │
│  ┌────────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐  │
│  │  Hierarchical   │  │     Sensor      │  │      GNN        │  │
│  │    Policy       │  │    Fusion       │  │  Communication  │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
└───────────┼─────────────────────┼─────────────────────┼─────────┘
            │                     │                     │
┌───────────┼─────────────────────┼─────────────────────┼─────────┐
│           │        ENVIRONMENT LAYER                  │         │
│  ┌────────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐  │
│  │   Swarm Env     │  │    Physics      │  │   Projectile    │  │
│  │  (PettingZoo)   │  │    Models       │  │    System       │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Layer Descriptions

### Environment Layer (Foundation)

The bottom layer provides the simulation foundation:

| Component              | File                      | Purpose                            |
| ---------------------- | ------------------------- | ---------------------------------- |
| **HypersonicSwarmEnv** | `hypersonic_swarm_env.py` | Core multi-agent environment       |
| **Physics Models**     | `physics_models.py`       | Atmospheric and trajectory physics |
| **Projectile System**  | `projectile_system.py`    | Guided interceptor missiles        |
| **Visualization**      | `visualization.py`        | Rendering and animation            |
| **RLlib Wrapper**      | `rllib_wrapper.py`        | Training framework adapter         |
| **Scaled Environment** | `scaled_environment.py`   | 50+ agent variant                  |

### Model Layer (Intelligence)

Neural networks and algorithms for perception and decision-making:

| Component                  | File                        | Purpose                               |
| -------------------------- | --------------------------- | ------------------------------------- |
| **Detection**              | `detection.py`              | Threat detection and classification   |
| **Adaptive Sensor Fusion** | `adaptive_sensor_fusion.py` | Advanced Kalman filtering             |
| **GNN Communication**      | `gnn_communication.py`      | Graph neural network coordination     |
| **Hierarchical Policy**    | `hierarchical_policy.py`    | Role-based decision making            |
| **ARSHI Integration**      | `arshi.py`                  | Resilient sensing & hive intelligence |
| **Opportunistic Sensors**  | `opportunistic_sensors.py`  | 6-modal degraded operations sensing   |
| **Belief System**          | `belief_system.py`          | Distributed Bayesian belief + gossip  |
| **Resilient GNN**          | `resilient_gnn.py`          | Degradation-aware communication       |

### Training Layer (Learning)

Reinforcement learning infrastructure:

| Component             | File                   | Purpose                        |
| --------------------- | ---------------------- | ------------------------------ |
| **MARL Training**     | `train_marl.py`        | RLlib PPO configuration        |
| **Curriculum**        | `curriculum.py`        | Progressive difficulty scaling |
| **Intrinsic Rewards** | `intrinsic_rewards.py` | Exploration bonuses            |
| **MAPPO**             | `mappo.py`             | Multi-agent PPO variant        |

### Application Layer (Interface)

User-facing tools and analysis:

| Component         | File               | Purpose                   |
| ----------------- | ------------------ | ------------------------- |
| **Dashboard**     | `app.py`           | Interactive web interface |
| **Metrics**       | `metrics.py`       | Performance tracking      |
| **Config Loader** | `config_loader.py` | YAML configuration        |
| **Logger**        | `logger.py`        | Structured logging        |

---

## Data Flow

### Training Flow

```
1. Configuration loaded from config.yaml
                ↓
2. RLlib creates training workers
                ↓
3. Each worker runs RLlibHyperionEnv (wrapper)
                ↓
4. Wrapper manages PettingZoo environment lifecycle
                ↓
5. Environment generates observations (25-dim per agent)
                ↓
6. Policy network produces actions (thrust, heading)
                ↓
7. Environment steps: physics, projectiles, detection
                ↓
8. Rewards computed (intercept/escape/shaping)
                ↓
9. Experiences collected → PPO updates
                ↓
10. Checkpoints saved periodically
```

### Inference Flow

```
1. Load trained policy from checkpoint
                ↓
2. Create environment instance
                ↓
3. For each timestep:
   ├─ Get observations
   ├─ Query policy → actions
   ├─ Step environment
   ├─ Check termination
   └─ Collect metrics
                ↓
4. Aggregate and export results
```

---

## ARSHI Resilient Operations Layer

For contested environments where radar/GPS are denied, ARSHI provides an additional intelligence layer:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ARSHI RESILIENT OPERATIONS                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │           OPPORTUNISTIC SENSOR SUITE (Layer 1)              │    │
│  │  Plasma | Acoustic | Thermal | Magnetic | PCL | Proprioception   │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │           BAYESIAN BELIEF FUSION (Layer 2)                  │    │
│  │     Particle Filter / Gaussian with uncertainty tracking    │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │           GOSSIP PROTOCOL (Layer 3)                         │    │
│  │     Epidemic belief sharing - works with 90%+ packet loss   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │           RESILIENT GNN (Layer 4)                           │    │
│  │     Degradation-aware coordination with isolation detection │    │
│  └─────────────────────────────────────────────────────────────┘    │ 
│                                                                     │
│  Operating Modes: FULL → DEGRADED → MINIMAL → PROPRIOCEPTIVE → ISOLATED  │
└─────────────────────────────────────────────────────────────────────┘
```

See [ARSHI System](ARSHI-System) for detailed documentation.

---

## Module Dependencies

```
utils/
├── config_loader.py (standalone)
└── logger.py (standalone)

env/
├── physics_models.py (standalone)
├── projectile_system.py → physics_models
├── hypersonic_swarm_env.py → physics_models, projectile_system
├── visualization.py → hypersonic_swarm_env
├── rllib_wrapper.py → hypersonic_swarm_env
└── scaled_environment.py → hypersonic_swarm_env

models/
├── detection.py → numpy, torch
├── adaptive_sensor_fusion.py → numpy
├── gnn_communication.py → torch, torch_geometric
├── hierarchical_policy.py → torch, gnn_communication
├── opportunistic_sensors.py → numpy, torch (ARSHI)
├── belief_system.py → numpy, scipy (ARSHI)
├── resilient_gnn.py → torch, torch_geometric (ARSHI)
└── arshi.py → opportunistic_sensors, belief_system, resilient_gnn (ARSHI)

training/
├── curriculum.py → env/hypersonic_swarm_env
├── intrinsic_rewards.py → numpy
├── mappo.py → torch, rllib
├── train_marl.py → rllib, env/rllib_wrapper
└── train_enhanced.py → train_marl, hierarchical_policy

evaluation/
└── metrics.py → env/hypersonic_swarm_env

dashboard/
└── app.py → streamlit, evaluation/metrics, env/*
```

---

## Design Principles

### 1. Separation of Concerns

Each layer has a single responsibility:
- Environment: Simulation physics
- Models: Intelligence/learning
- Training: Optimization
- Application: User interface

### 2. Dependency Inversion

Higher layers depend on abstractions:
- Training uses `MultiAgentEnv` interface, not concrete environment
- Dashboard uses `EvaluationMetrics`, not raw data structures

### 3. Configuration over Code

Hyperparameters in YAML, not hardcoded:
- Easy experimentation
- Reproducible experiments
- No code changes for parameter sweeps

### 4. Testability

Each module is independently testable:
- Unit tests for components
- Integration tests for pipelines
- End-to-end tests for full episodes

---

## Scalability Architecture

### Horizontal Scaling (More Machines)

```
┌─────────────────────────────────────────┐
│              Ray Cluster                │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐    │
│  │ Worker 1│ │ Worker 2│ │ Worker N│    │
│  │  Env    │ │  Env    │ │  Env    │    │
│  │ Rollout │ │ Rollout │ │ Rollout │    │
│  └────┬────┘ └────┬────┘ └────┬────┘    │
│       │           │           │         │
│       └───────────┼───────────┘         │
│                   ▼                     │
│           ┌─────────────┐               │
│           │   Learner   │               │
│           │ (PPO Update)│               │
│           └─────────────┘               │
└─────────────────────────────────────────┘
```

### Vertical Scaling (Bigger Machines)

- **GPU Acceleration**: Policy networks on CUDA
- **Vectorized Environments**: Batch physics updates
- **Efficient NumPy**: Optimized array operations

### Agent Scaling (More Agents)

The `ScaledEnvironment` supports 50-100+ agents through:
- Graph Neural Networks for O(n) communication (vs O(n²))
- Hierarchical policies for reduced action space
- Efficient k-nearest neighbor algorithms

---

## Security Considerations

| Aspect                 | Implementation                                         |
| ---------------------- | ------------------------------------------------------ |
| No Real Weapons        | Simulation only, no hardware interfaces                |
| Data Privacy           | Federated learning capability for classified scenarios |
| Model Security         | Checkpoint encryption possible                         |
| Adversarial Robustness | Red team training for robust policies                  |
