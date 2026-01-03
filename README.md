<br>

<p align="center">
<img width="600"  alt="HYPERION" src="https://github.com/user-attachments/assets/f24655b8-5b67-4dfd-b9bc-9175182c3817" />
<p align="center">
  <strong>AI-Driven Multi-Agent Swarm Intelligence for Hypersonic Defense</strong><br>
</p>  
<br>

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=for-the-badge)](https://www.python.org/downloads/)
[![GPLv3.0](https://img.shields.io/badge/GPL--3.0-red?style=for-the-badge)](https://choosealicense.com/licenses/gpl-3.0/)

HYPERION is a sophisticated machine learning platform designed to simulate and optimize autonomous drone swarms for the detection, tracking, and interception of hypersonic and sub hypersonic threats in aerospace and defense scenarios. The system leverages Multi-Agent Reinforcement Learning (MARL) to enable decentralized coordination among UAVs in high-stakes, dynamic environments.

## Key Features

- **Multi-Agent Reinforcement Learning**: Decentralized swarm coordination using PPO and RLlib
- **ARSHI System**: Autonomous Resilient Sensing & Hive Intelligence for contested environments
- **Opportunistic Sensing**: 6 unconventional sensor modalities for degraded operations
- **Swarm Proprioception**: Novel behavioral inference when sensors are denied
- **Gossip-Based Belief Propagation**: Convergent consensus with 90%+ packet loss tolerance
- **Advanced Sensor Fusion**: Multi-modal detection combining RF fingerprinting and thermal signatures
- **Realistic Physics Simulation**: Hypersonic trajectories (Mach 5+), atmospheric modeling, fuel dynamics
- **Production-Ready Pipeline**: Complete training, evaluation, and deployment infrastructure
- **Interactive Dashboard**: Real-time visualization and performance analysis with Streamlit
- **Comprehensive Metrics**: Interception rates, fuel efficiency, coordination quality tracking

## Strategic Relevance

Hypersonic weapons represent a critical challenge in modern defense, with nations investing heavily in countermeasures. HYPERION addresses this by:

- Aligning with DoD priorities in multi-domain operations and resilient autonomy
- Demonstrating advanced ML techniques (MARL, GNNs, sensor fusion)
- Providing measurable outcomes (>85% interception target in simulations)
- Including adversarial robustness and explainable AI for human oversight

## Installation

### Prerequisites

- Python 3.10 or higher
- Conda (recommended) or pip
- 16GB+ RAM for training

### Setup
```bash
# Clone repository
git clone https://github.com/yourusername/hyperion.git
cd hyperion

# Create conda environment
conda create -n hyperion python=3.10 -y
conda activate hyperion

# Install dependencies
conda-forge install -r requirements.txt
# or pip install -r requirements.txt
``` 

## Quick Start

### 1. Test Environment
```bash
# Run environment tests
python tests/test_environment.py

# Test visualization
python tests/test_visualization.py
```

### 2. Launch Dashboard
```bash
# Start interactive dashboard
streamlit run src/dashboard/app.py
```
The dashboard provides:
- Live swarm simulation with pursuit policy
- Multi-episode evaluation
- Performance metrics and analysis
- Saved results visualization

### 3. Run Training (Optional)
```bash
# Train swarm coordination policy
python src/training/train_marl.py --iterations 100

# Or use custom config
python src/training/train_marl.py --config config.yaml --iterations 500
```

### 4. Evaluate Policy
```bash
# Evaluate trained policy
python -c "
from src.env.hypersonic_swarm_env import HypersonicSwarmEnv
from src.evaluation.metrics import evaluate_policy

env = HypersonicSwarmEnv(num_agents=5)
metrics = evaluate_policy(env, policy=None, num_episodes=100)
"
```

## Project Structure
```
hyperion/
├── src/
│   ├── env/                    # Environment implementation
│   │   ├── hypersonic_swarm_env.py
│   │   ├── scaled_environment.py
│   │   ├── physics_models.py
│   │   ├── projectile_system.py
│   │   ├── visualization.py
│   │   └── rllib_wrapper.py
│   ├── models/                 # ML models
│   │   ├── arshi.py            # ARSHI integration module
│   │   ├── opportunistic_sensors.py  # Multi-modal degraded ops sensors
│   │   ├── belief_system.py    # Distributed Bayesian belief + gossip
│   │   ├── resilient_gnn.py    # Degradation-aware GNN communication
│   │   ├── gnn_communication.py
│   │   ├── hierarchical_policy.py
│   │   ├── detection.py        # Sensor fusion and threat detection
│   │   └── adaptive_sensor_fusion.py
│   ├── training/               # Training pipeline
│   │   ├── train_marl.py
│   │   ├── train_enhanced.py
│   │   └── curriculum.py
│   ├── evaluation/             # Evaluation metrics
│   │   └── metrics.py
│   ├── dashboard/              # Streamlit dashboard
│   │   └── app.py
│   └── utils/                  # Utilities
│       ├── config_loader.py
│       └── logger.py
├── tests/                      # Test suite (90%+ coverage)
├── outputs/                    # Generated outputs
├── checkpoints/                # Model checkpoints
├── config.yaml                 # Configuration
├── requirements.txt
└── README.md
```

## Core Components

### 1. Environment (PettingZoo Parallel API)

Multi-agent environment with:
- **Agents**: 5-10 UAVs with fuel, sensors, interceptors
- **Target**: Hypersonic vehicle (Mach 5+) with ballistic trajectory
- **Observation Space**: Own state, target detection, neighbor positions (25 dims)
- **Action Space**: Thrust magnitude [0,1], heading change [-1,1]
- **Rewards**: +100 interception, -100 escape, distance/fuel penalties

### 2. ARSHI: Autonomous Resilient Sensing & Hive Intelligence

The ARSHI system enables continued operation in contested electromagnetic environments where traditional sensors and communications are degraded or denied.

**Opportunistic Sensor Suite** (6 unconventional modalities):
| Sensor                    | Signal Source                  | Jamming Immunity |
| ------------------------- | ------------------------------ | ---------------- |
| Plasma Emission           | Hypersonic plasma sheath RF    | High             |
| Acoustic Array            | Sonic boom triangulation       | Complete         |
| Thermal Wake              | Atmospheric heating signatures | High             |
| Magnetic Anomaly          | Metallic mass perturbations    | Complete         |
| Passive Coherent Location | Ambient RF (cell/FM) scatter   | Medium           |
| Swarm Proprioception      | Neighbor behavioral inference  | Complete         |

**Distributed Belief System**:
- Particle filter and Gaussian belief representations
- Gossip-based belief propagation (works with 90%+ packet loss)
- Covariance intersection for unknown correlations
- Automatic belief decay for stale information

**Operating Modes**:
```
FULL → DEGRADED → MINIMAL → PROPRIOCEPTIVE → ISOLATED
```
Automatic mode switching based on sensor/communication availability.

### 3. Detection Module

Multi-sensor fusion:
- Kalman filtering for position tracking
- Neural network threat classifier
- Multi-target tracking with data association
- RF and thermal sensor simulation

### 4. Training Pipeline

RLlib-based MARL:
- PPO algorithm with shared policy
- Curriculum learning (subsonic → hypersonic)
- Distributed training with Ray
- Automatic checkpointing

### 5. Evaluation System

Comprehensive metrics:
- Interception rate
- Episode length and rewards
- Fuel efficiency
- Minimum distance to target
- JSON export for analysis

## Configuration

Edit `config.yaml` to customize:
```yaml
environment:
  num_agents: 5
  target_speed: 1700.0  # m/s (Mach 5)
  detection_range: 2000.0
  
training:
  algorithm: "PPO"
  num_workers: 4
  num_gpus: 0
  
curriculum:
  enabled: true
  stages:
    - name: "basic"
      target_speed: 500.0
      duration_episodes: 1000
```

## Performance Benchmarks

Random policy baseline (10 episodes):
- Interception Rate: ~0-10%
- Mean Episode Length: ~200 steps
- Mean Episode Reward: -50 to -100

Training targets:
- Interception Rate: >85%
- Decision Latency: <100ms
- GPS-denied operation: Functional

## Testing
```bash
# Run all tests
pytest tests/ -v

# Or run individual test modules
python -m pytest tests/test_environment.py
python -m pytest tests/test_arshi.py      # ARSHI resilient sensing (33 tests)
python -m pytest tests/test_detection.py
python -m pytest tests/test_training_setup.py
python -m pytest tests/test_evaluation.py
python -m pytest tests/test_visualization.py
```

## Technical Details

### ARSHI Architecture

![Hive Intelligence Architecture drawio](https://github.com/user-attachments/assets/64b3b59b-59fb-42d9-bf9a-885bf51fa3a3)


### Physics Model

- Constant velocity model for hypersonic target
- Euler integration (dt=0.1s)
- Max agent speed: 300 m/s
- Max target speed: 1700 m/s (Mach 5)
- Atmospheric drag (optional via physics_models.py)

### Sensor Model

- Detection range: 2000m
- Communication range: 1500m
- Intercept range: 50m
- Noisy measurements with confidence weighting

### Reward Function
```python
if intercepted:
    reward = +100
elif escaped:
    reward = -100
else:
    reward = -distance_penalty - fuel_penalty + formation_bonus
```

## Future Enhancements

### Phase 2: Advanced ML

- [X] Graph Neural Networks for agent communication
- [X] MAPPO or QMIX for value decomposition
- [X] Attention mechanisms for dynamic graphs
- [X] Adversarial training (red team evasion)

### Phase 3: Resilient Operations (ARSHI)

- [X] Opportunistic multi-modal sensing (plasma, acoustic, thermal, magnetic, PCL)
- [X] Swarm proprioception (behavioral inference)
- [X] Gossip-based belief propagation
- [X] Degradation-aware GNN communication
- [X] Automatic mode switching (FULL → ISOLATED)
- [ ] 3D environment
- [ ] Evasive target maneuvers
- [ ] Hardware-in-the-loop testing

### Phase 4: Deployment

- [ ] Edge AI optimization (quantization)
- [ ] Real-time constraints (<100ms)
- [ ] Kubernetes deployment
- [ ] Digital twin integration
- [ ] Federated learning for secure training

## Applications

- Defense contractor demonstrations
- DARPA research proposals
- Autonomous systems testing
- Multi-agent coordination research
- Aerospace simulation platforms

## License

GNU GENERAL PUBLIC LICENSE - see LICENSE file for details

## Acknowledgments

Built on:
- PettingZoo (multi-agent environments)
- RLlib (reinforcement learning)
- PyTorch (neural networks)
- Streamlit (dashboard)

Inspired by prior portfolio projects:
- SENTINEL: Multi-Intelligence early warning platform
- CONSTELLATION: Satellite fleet health management
- MERIDIAN: Portfolio optimization system

## Contributing

This is a portfolio project. For questions or collaboration:

[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](michaelgurule1164@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](www.linkedin.com/in/michael-j-gurule-447aa2134)
[![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@michaelgurule1164)

---

<p align="center">
  <img src="https://github.com/user-attachments/assets/0d69bf96-335b-4160-a202-780e8bad2d45" alt="MICHAEL GURULE">
</p>
<p align="center"><sub>HYPERION | Data: (Public)</sub>
</p>
