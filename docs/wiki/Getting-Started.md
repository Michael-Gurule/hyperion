# Getting Started

This guide will help you get HYPERION up and running quickly.

## Prerequisites

- **Python 3.10+**: Required for type hints and modern features
- **16GB+ RAM**: Training requires significant memory for parallel environments
- **Conda** (recommended): Simplifies dependency management

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/hyperion.git
cd hyperion
```

### 2. Create Environment

```bash
# Using Conda (recommended)
conda create -n hyperion python=3.10 -y
conda activate hyperion

# Or using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 3. Install Dependencies

```bash
conda-forge install -r requirments.txt
# or pip install -r requirements.txt
```

### 4. Verify Installation

```bash
# Run environment tests
python tests/test_environment.py

# Run all tests
python -m pytest tests/ -v
```

---

## Quick Demo

### Option 1: Interactive Dashboard

The fastest way to see HYPERION in action:

```bash
streamlit run src/dashboard/app.py
```

This launches a web interface where you can:
- Watch live swarm simulations
- Run batch evaluations
- Analyze performance metrics
- Visualize trajectories

### Option 2: Python Script

```python
from src.env.hypersonic_swarm_env import HypersonicSwarmEnv
from src.env.visualization import SwarmVisualizer

# Create environment
env = HypersonicSwarmEnv(num_agents=5)
visualizer = SwarmVisualizer(env)

# Run episode with random policy
observations, infos = env.reset()

for step in range(200):
    # Random actions for demonstration
    actions = {agent: env.action_space(agent).sample()
               for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)
    visualizer.render()

    if all(terminations.values()):
        break

env.close()
```

### Option 3: Jupyter Notebook

See [`notebooks/physics_walkthrough.ipynb`](../../notebooks/physics_walkthrough.ipynb) for an interactive exploration of the physics models.

---

## Project Structure

```
hyperion/
├── src/
│   ├── env/                    # Simulation environment
│   ├── models/                 # ML models (detection, policies)
│   ├── training/               # Training pipeline
│   ├── evaluation/             # Metrics and analysis
│   ├── dashboard/              # Web interface
│   └── utils/                  # Configuration, logging
├── tests/                      # Test suite
├── notebooks/                  # Jupyter notebooks
├── docs/                       # Documentation
├── checkpoints/                # Saved model weights
├── outputs/                    # Generated results
└── config.yaml                 # Master configuration
```

---

## Configuration

All hyperparameters are centralized in `config.yaml`:

```yaml
environment:
  num_agents: 5
  target_speed: 1700.0      # m/s (Mach 5)
  detection_range: 2000.0   # meters
  communication_range: 1500.0

training:
  algorithm: "PPO"
  learning_rate: 0.0003
  num_workers: 4
  train_batch_size: 4000

curriculum:
  enabled: true
  stages:
    - name: "basic"
      target_speed: 500.0
      duration_episodes: 1000
    - name: "intermediate"
      target_speed: 1000.0
      duration_episodes: 2000
    - name: "advanced"
      target_speed: 1700.0
```

---
