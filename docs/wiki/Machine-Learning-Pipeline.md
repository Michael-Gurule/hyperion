# Machine Learning Pipeline

This page details the reinforcement learning algorithms, training procedures, and neural network architectures used in HYPERION.

---

## Overview

HYPERION uses Multi-Agent Reinforcement Learning (MARL) to train cooperative policies for UAV swarms. The key challenge is coordinating multiple agents with:

- **Partial observability**: Each agent sees only nearby entities
- **Decentralized execution**: No central controller during deployment
- **Sparse rewards**: Success/failure determined at episode end
- **Credit assignment**: Which agent's actions led to success?

---

## 1. Algorithm: PPO (Proximal Policy Optimization)

### Why PPO?

| Property           | Benefit for HYPERION                            |
| ------------------ | ----------------------------------------------- |
| Sample efficient   | Fewer environment steps needed                  |
| Stable training    | Clipped objective prevents catastrophic updates |
| On-policy          | Fresh experience, no replay buffer complexity   |
| Continuous actions | Natural fit for thrust/heading control          |

### PPO Objective

```
L^CLIP(θ) = E[min(r_t(θ)·A_t, clip(r_t(θ), 1-ε, 1+ε)·A_t)]
```

Where:
- r_t(θ) = π_θ(a|s) / π_θ_old(a|s) (probability ratio)
- A_t = advantage estimate (GAE)
- ε = clip parameter (0.2)

### Hyperparameters

```yaml
learning_rate: 0.0003
gamma: 0.99              # Discount factor
gae_lambda: 0.95         # GAE parameter
clip_param: 0.2          # PPO clip
entropy_coeff: 0.01      # Exploration bonus
vf_coeff: 0.5            # Value function loss weight
num_sgd_iter: 10         # SGD epochs per batch
minibatch_size: 128
train_batch_size: 4000   # Steps per training iteration
```

---

## 2. Multi-Agent Configuration

### Shared Policy

All agents share a single policy network:

```
Observation (Agent 1) ──┐
                        │
Observation (Agent 2) ──┼──→ Shared Policy ──→ Actions
                        │      Network
Observation (Agent N) ──┘
```

**Advantages**:
- Reduced sample complexity (N× more experience per update)
- Natural homogeneity (all UAVs have same capabilities)
- Simpler deployment (single model)

**Disadvantages**:
- Can't learn specialized roles (addressed by hierarchical policy)
- May converge to symmetric behaviors

### Centralized Training, Decentralized Execution (CTDE)

During training:
- Global information available for value function
- Reward shaping uses global state

During execution:
- Each agent uses only local observations
- No communication with central server required

---

## 3. Policy Network Architecture

### Standard Policy

```
Input: Observation (25 dims)
    ↓
FC(25 → 256) + ReLU
    ↓
FC(256 → 256) + ReLU
    ↓
┌───────────────────────────┐
│                           │
↓                           ↓
FC(256 → 2)            FC(256 → 2)
Action Mean            Action Log Std
    ↓                       ↓
    └───────┬───────────────┘
            ↓
    Gaussian Distribution
            ↓
        Sample Action
```

### Hierarchical Policy (Advanced)

```
                    Observation
                         ↓
                  ┌──────────────┐
                  │  GNN Encoder │
                  │  (Message    │
                  │   Passing)   │
                  └──────┬───────┘
                         ↓
                  Agent Embedding
                         ↓
              ┌──────────┴──────────┐
              ↓                     ↓
        Role Assigner          Role Policies
        (Every K steps)        (Per-timestep)
              ↓                     ↓
        Role Selection    Role-Conditioned Action
              │                     ↑
              └─────────────────────┘
```

### GNN Communication Layer

Graph attention for agent coordination:

```python
# Node features: Agent observations
# Edges: Agents within communication range

h_i^(l+1) = σ(Σ_j α_ij · W · h_j^(l))

# Attention weights
α_ij = softmax(LeakyReLU(a^T · [W·h_i || W·h_j || e_ij]))

# Edge features e_ij: relative position, distance
```

---

## 4. Reward Engineering

### Extrinsic Rewards

```python
# Terminal rewards
if target_intercepted:
    reward = +100.0
elif target_escaped:
    reward = -100.0
```

### Reward Shaping

```python
# Distance-based (encourages approach)
distance_reward = -0.01 * distance_to_target

# Fuel penalty (encourages efficiency)
fuel_penalty = -0.1 * fuel_consumed

# Formation bonus (encourages coordination)
formation_reward = +0.1 * formation_quality

# Total shaped reward
reward = distance_reward + fuel_penalty + formation_reward
```

### Intrinsic Rewards

Additional exploration bonuses:

| Reward Type        | Purpose                        |
| ------------------ | ------------------------------ |
| Novelty            | Encourage visiting new states  |
| Velocity mismatch  | Penalize passive following     |
| Intercept geometry | Reward optimal approach angles |
| Detection coverage | Reward spreading out           |

```python
# Velocity mismatch penalty
velocity_alignment = dot(agent_vel, target_vel) / (|agent_vel| * |target_vel|)
mismatch_bonus = -0.05 * velocity_alignment  # Penalize parallel trajectories
```

---

## 5. Curriculum Learning

### Motivation

Hypersonic interception is extremely difficult:
- Target moves 5-6× faster than interceptors
- Requires precise coordination
- Sparse success signal

Starting with the full problem leads to:
- No learning signal (always fails)
- Random policy convergence
- Unstable training

### Curriculum Stages

| Stage        | Target Speed | Evasion | Agents | Episodes          |
| ------------ | ------------ | ------- | ------ | ----------------- |
| Basic        | 500 m/s      | None    | 3      | 1,000             |
| Intermediate | 1,000 m/s    | Basic   | 5      | 2,000             |
| Advanced     | 1,700 m/s    | Full    | 5      | Until convergence |

### Advancement Criteria

```python
# Track rolling success rate
success_rate = rolling_mean(successes, window=50)

# Advance if consistently successful
if success_rate > 0.7 and episodes_in_stage > min_episodes:
    advance_to_next_stage()

# Regress if struggling (optional)
if success_rate < 0.3 and episodes_in_stage > patience:
    regress_to_previous_stage()
```

### Implementation

```python
from src.training.curriculum import CurriculumScheduler

scheduler = CurriculumScheduler(
    stages=[
        CurriculumStage("basic", speed=500, evasion=None),
        CurriculumStage("intermediate", speed=1000, evasion="basic"),
        CurriculumStage("advanced", speed=1700, evasion="full"),
    ],
    success_threshold=0.7,
    min_episodes=100
)

# Training loop
for episode in range(total_episodes):
    config = scheduler.get_current_config()
    env.configure(config)

    success = run_episode(env, policy)
    scheduler.record_episode(success)

    if scheduler.should_advance():
        scheduler.advance_stage()
```

---

## 6. Training Infrastructure

### Distributed Training with Ray

```
┌─────────────────────────────────────────────────────────────┐
│                        Ray Cluster                          │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │  Rollout    │  │  Rollout    │  │  Rollout    │  ...     │
│  │  Worker 1   │  │  Worker 2   │  │  Worker N   │          │
│  │  ┌───────┐  │  │  ┌───────┐  │  │  ┌───────┐  │          │
│  │  │  Env  │  │  │  │  Env  │  │  │  │  Env  │  │          │
│  │  └───────┘  │  │  └───────┘  │  │  └───────┘  │          │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          │
│         │                │                │                 │
│         └────────────────┼────────────────┘                 │
│                          ↓                                  │
│                  ┌───────────────┐                          │
│                  │    Learner    │                          │
│                  │  (PPO Update) │                          │
│                  │  ┌─────────┐  │                          │
│                  │  │  GPU    │  │                          │
│                  │  │ (opt.)  │  │                          │
│                  │  └─────────┘  │                          │
│                  └───────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

### Configuration

```python
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment(
        env="hyperion",
        env_config={"num_agents": 5}
    )
    .training(
        lr=3e-4,
        gamma=0.99,
        train_batch_size=4000,
        sgd_minibatch_size=128,
        num_sgd_iter=10
    )
    .rollouts(
        num_rollout_workers=4,
        rollout_fragment_length=200
    )
    .resources(
        num_gpus=0  # or 1 for GPU training
    )
    .multi_agent(
        policies={"shared_policy": PolicySpec()},
        policy_mapping_fn=lambda agent_id, *args: "shared_policy"
    )
)
```

### Checkpointing

```python
# Automatic checkpointing
trainer.save("checkpoints/")

# Load checkpoint
trainer.restore("checkpoints/checkpoint_000100")

# Export for deployment
policy = trainer.get_policy()
policy.export_model("models/hyperion_policy")
```

---

## 7. Evaluation Protocol

### Metrics Tracked

| Metric            | Description               | Target   |
| ----------------- | ------------------------- | -------- |
| Interception rate | % of successful episodes  | >85%     |
| Mean reward       | Average episode return    | >50      |
| Episode length    | Steps to termination      | Minimize |
| Fuel efficiency   | Fuel remaining at success | >50%     |
| Time to intercept | Steps until interception  | Minimize |

### Evaluation Script

```python
from src.evaluation.metrics import EvaluationMetrics

metrics = EvaluationMetrics()

for episode in range(100):
    obs, _ = env.reset()
    done = False
    episode_reward = 0

    while not done:
        actions = {agent: policy.compute_action(obs[agent])
                   for agent in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        episode_reward += sum(rewards.values())
        done = all(terms.values())

    metrics.add_episode(
        reward=episode_reward,
        length=env.step_count,
        success="intercept" in infos.get("termination_reason", "")
    )

summary = metrics.get_summary()
print(f"Interception Rate: {summary['interception_rate']:.1%}")
```

---

## 8. Advanced Techniques

### MAPPO (Multi-Agent PPO)

Centralized value function with decentralized policies:

```
V(s) = V(o_1, o_2, ..., o_N)  # Global state
π_i(a_i|o_i)                   # Local policy
```

### Value Decomposition (QMIX-style)

```
Q_tot = f(Q_1, Q_2, ..., Q_N)

# Monotonic mixing
∂Q_tot/∂Q_i ≥ 0
```

### Attention Mechanisms

Dynamic focus on relevant agents:

```python
attention_weights = softmax(Q @ K.T / sqrt(d_k))
context = attention_weights @ V
```

---

## 9. Debugging & Monitoring

### TensorBoard Integration

```bash
tensorboard --logdir=~/ray_results/
```

Tracked metrics:
- Episode reward (mean, min, max)
- Policy entropy
- Value function loss
- KL divergence
- Curriculum stage

### Common Issues

| Issue             | Symptom                  | Solution                             |
| ----------------- | ------------------------ | ------------------------------------ |
| No learning       | Flat reward curve        | Check reward scale, curriculum       |
| Unstable training | Oscillating rewards      | Reduce learning rate, increase batch |
| Policy collapse   | All agents same action   | Add entropy bonus                    |
| Slow learning     | Very gradual improvement | Increase workers, batch size         |

### Debugging Checklist

1. **Environment**: Test with random policy, verify rewards
2. **Observations**: Check normalization, NaN values
3. **Actions**: Verify action bounds, clipping
4. **Rewards**: Print reward components each step
5. **Termination**: Confirm episodes end correctly
