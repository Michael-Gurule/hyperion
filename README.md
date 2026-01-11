<br>

<p align="center">
<img width="600" alt="HYPERION" src="https://github.com/user-attachments/assets/6ae91cdd-dd78-403f-a506-df7f2e51f669" />
<p align="center">
  <strong>AI-Driven Multi-Agent Swarm Intelligence for Hypersonic Defense</strong><br>
</p>  
<br>

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=for-the-badge)](https://www.python.org/downloads/)
[![GPLv3.0](https://img.shields.io/badge/GPL--3.0-red?style=for-the-badge)](https://choosealicense.com/licenses/gpl-3.0/)

HYPERION is a sophisticated Multi-Agent Reinforcement Learning (MARL) framework designed to simulate and optimize autonomous drone swarms for the detection, tracking, and interception of hypersonic and sub hypersonic threats in aerospace and defense scenarios. The system leverages Multi-Agent Reinforcement Learning (MARL) to enable decentralized coordination among UAVs in high-stakes, dynamic environments.

## 🏗️ MLOps Architecture (v2.0)

This project has been re-architected to follow industry-standard MLOps practices, ensuring reproducibility, scalability, and clean separation of concerns.

| Component | Tool | Description |
| :--- | :--- | :--- |
| **Configuration** | **Hydra** | Compositional, hierarchical config management. Experiments are reproducible via config snapshots. |
| **Tracking** | **Weights & Biases** | Live metrics logging (Reward, Loss, Success Rate), system monitoring, and model artifact versioning. |
| **Data Lineage** | **DVC** | Version control for large simulation assets (maps, physics tables) and training datasets. |
| **Orchestration** | **Python/Bash** | Modular training loop decoupled from environment logic and agent architecture. |

### Directory Structure

```text
HYPERION/
├── conf/                # Hydra configuration (Experiment definitions)
├── data/                # DVC-tracked assets
├── src/
│   ├── env/             # Physics engine & Scaled Environment
│   └── models/          # MARL Agents (MAPPO, Hierarchical), GNNs, Sensors
├── experiments/         # Lab notebook & hypothesis logs
├── train.py             # Main entry point (Hydra-wrapped)
└── ...

**🧪 Experimentation Workflow**

HYPERION follows a rigorous "Scientific Method" workflow for feature integration:

1. Baseline Establishment: A controlled run (Tag: baseline) establishes the "Control Group" performance.
2. Hypothesis Formulation: Features are developed to address specific failure modes (e.g., "Intrinsic rewards will reduce tail-chasing").
3. A/B Testing: The "Treatment Group" is trained with the new feature active.Verification: Success is measured via W&B metrics (success_rate, interception_efficiency).

| Feature Implementation Status                                                                             |
| Feature              | Status        | DescriptionScaled                                                  |
| -------------------- | ------------- | ------------------------------------------------------------------ |
|: Swarm Env           | ✅ Active     | 50-100 agent environment with ISA atmosphere & projectile physics  |
|: Curriculum Learning | ✅ Active     | 4-stage difficulty progression (Ballistic $\to$ Evasive)           |
|: MAPPO Agent         | ✅ Active     | Centralized Critic PPO for homogeneous swarms                      | 
|: Intrinsic Rewards   | ⚠️ In Testing | Anti-trailing penalties & novelty search (Exp: EXP-001)            |
|: Swarm GNN           | ⏳ Planned    | Graph Neural Networks for emergent communication                   | 
|: Sensor Fusion       | ⏳ Planned    | Kalman Filtering & Multi-modal signal processing                   | 
|: ARSHI Resilience    | ⏳ Planned    | Autonomous degradation handling under failure                      |



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

## Quick Start

### Setup

1. Installation
```
Bash
#Clone and Install Dependencies
git clone [https://github.com/michael-gurule/hyperion.git](https://github.com/michael-gurule/hyperion.git)
cd hyperion

conda-forge install -r requirements.txt
# or pip install -r requirements.txt
```

2. Run the Baseline (Control Group)
Launch a standard training run with 50 agents to verify system performance.

```bash
python train.py environment.num_agents=50 experiment_name="baseline-control"
```
3. Run an Experiment (Hydra Override)
Test a specific hypothesis (e.g., changing learning rate or enabling intrinsics) without modifying code.

``` 
train.py agent.hyperparameters.lr=0.0005 +experiment=hypersonic_swarm
```

📊 Results & Artifacts
All training metrics, system logs, and model checkpoints are automatically synced to the Weights & Biases Dashboard.
Baseline Success Rate: ~30% (Stage 1)
Target Interception: < 15s (Average)

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
- W&B Reports for analysis

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

Random policy baseline (X episodes):
- Interception Rate: 
- Mean Episode Length: 
- Mean Episode Reward:

Training targets:
- Interception Rate: 
- Decision Latency: 
- GPS-denied operation:

## System Testing
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

