# Design Decisions & Rationale

This page explains the key architectural and algorithmic decisions made during HYPERION's development, along with the reasoning behind each choice.

---

## 1. Environment Design

### Decision: PettingZoo + RLlib Combination

**Choice**: Use PettingZoo for environment API, RLlib for training.

**Alternatives Considered**:
- OpenAI Gym (single-agent only)
- Custom environment (no ecosystem)
- Unity ML-Agents (overkill for 2D)

**Rationale**:
- PettingZoo is the standard for multi-agent environments
- RLlib provides production-grade distributed training
- Clean separation: environment logic vs. training infrastructure
- Both are actively maintained with strong communities

**Trade-off**: Requires wrapper layer, but gains ecosystem compatibility.

---

### Decision: 2D Environment (Not 3D)

**Choice**: Implement 2D dynamics initially.

**Alternatives Considered**:
- Full 3D environment
- 2.5D (3D positions, 2D actions)

**Rationale**:
- Captures essential coordination dynamics
- Faster training iterations (simpler physics)
- Easier visualization and debugging
- 3D adds complexity without proportional insight gain initially

**Trade-off**: Less realistic, but enables rapid prototyping. 3D is planned for Phase 3.

---

### Decision: Continuous Action Space

**Choice**: Continuous [thrust, heading_change] actions.

**Alternatives Considered**:
- Discrete actions (8 directions + thrust levels)
- Hybrid (discrete heading, continuous thrust)

**Rationale**:
- More natural for aerospace control
- Finer-grained maneuvering
- PPO handles continuous actions well
- Matches real actuator interfaces

**Trade-off**: Harder exploration than discrete, but more expressive.

---

## 2. Multi-Agent Learning

### Decision: Shared Policy Across Agents

**Choice**: All agents use identical policy network.

**Alternatives Considered**:
- Independent policies per agent
- Heterogeneous agent types
- Parameter sharing with agent ID

**Rationale**:
- All UAVs have identical capabilities (homogeneous swarm)
- N× sample efficiency (every agent generates training data)
- Simpler deployment (single model)
- Natural scalability (works for any number of agents)

**Trade-off**: Can't learn specialized roles without additional mechanism (hence hierarchical policy).

---

### Decision: Hierarchical Policy for Role Specialization

**Choice**: Add role-based option-critic architecture.

**Problem Addressed**: Shared policy leads to symmetric behaviors; we need scouts, trackers, interceptors.

**Design**:
```
Manager (Role Assigner) → Assigns roles every K steps
Workers (Role Policies) → Execute role-specific behaviors
```

**Rationale**:
- Enables specialization within shared policy framework
- Roles are interpretable (human-understandable)
- Reduces exploration space (4 roles vs. infinite behaviors)
- Manager learns when to reassign roles

**Trade-off**: More complex architecture, but gains strategic diversity.

---

### Decision: GNN for Agent Communication

**Choice**: Graph Neural Networks for coordination.

**Alternatives Considered**:
- Full attention (all-to-all)
- Fixed communication topology
- No explicit communication (implicit through observation)

**Rationale**:
- O(E) complexity vs O(N²) for full attention
- Dynamic graph based on communication range
- Permutation invariant (works for any agent ordering)
- Attention mechanisms focus on relevant neighbors

**Trade-off**: Requires PyTorch Geometric dependency, but scales to 100+ agents.

---

## 3. Projectile System

### Decision: Proportional Navigation Guidance

**Choice**: Use PN guidance for interceptor missiles.

**Alternatives Considered**:
- Pursuit guidance (point-and-chase)
- Predicted intercept point (PIP)
- Learned guidance (neural network)

**Rationale**:
- PN is the industry standard for missile guidance
- Mathematically optimal for constant-velocity targets
- Robust to moderate target maneuvers
- Simple to implement and debug
- Well-understood failure modes

**Trade-off**: Not optimal for highly maneuvering targets, but handles our scenarios well.

---

### Decision: Projectiles Launched by Agents (Not Environment)

**Choice**: Agents control when to fire projectiles.

**Alternatives Considered**:
- Automatic firing when in range
- Central controller decides firing
- No projectiles (direct UAV interception)

**Rationale**:
- Bridges speed gap (UAV: 300 m/s, Target: 1700 m/s, Projectile: 600 m/s)
- Adds tactical decision layer (when to fire?)
- More realistic engagement model
- Creates interesting credit assignment problem

**Trade-off**: Increases action complexity, but necessary for feasible interception.

---

## 4. Physics Modeling

### Decision: ISA Atmosphere Model

**Choice**: Implement International Standard Atmosphere.

**Alternatives Considered**:
- Constant atmospheric properties
- Full weather modeling
- Lookup tables from real data

**Rationale**:
- Industry standard for aerospace simulation
- Captures altitude effects on drag and sensors
- Computationally cheap (analytical formulas)
- Enables altitude-based strategies

**Trade-off**: Ignores weather, but provides essential altitude physics.

---

### Decision: Euler Integration (Not Higher-Order)

**Choice**: Simple Euler method for physics integration.

**Alternatives Considered**:
- RK4 (Runge-Kutta 4th order)
- Verlet integration
- Analytical solutions where possible

**Rationale**:
- Sufficient accuracy for dt=0.1s timesteps
- Fastest computation per step
- Easy to understand and debug
- Errors are systematic, not random (RL can adapt)

**Trade-off**: Lower accuracy than RK4, but acceptable for RL training.

---

## 5. Sensor Modeling

### Decision: Probabilistic Detection Model

**Choice**: Detection is probabilistic, not deterministic.

**Alternatives Considered**:
- Binary detection (in-range = detected)
- Perfect detection everywhere
- Complex radar equation modeling

**Rationale**:
- Matches real sensor behavior
- Forces agents to handle uncertainty
- Quadratic falloff models radar physics
- Encourages approach for reliable detection

**Trade-off**: Adds stochasticity, but more realistic.

---

### Decision: Kalman Filter for Sensor Fusion

**Choice**: Standard Kalman filter, not advanced variants.

**Alternatives Considered**:
- Extended Kalman Filter (EKF)
- Unscented Kalman Filter (UKF)
- Particle Filter
- Neural network fusion

**Rationale**:
- Optimal for linear Gaussian systems
- Our dynamics are approximately linear
- Well-understood, easy to debug
- Provides uncertainty estimates

**Trade-off**: Suboptimal for nonlinear dynamics, but sufficient here.

---

## 6. Training Strategy

### Decision: Curriculum Learning with 3 Stages

**Choice**: Progressive difficulty from subsonic to hypersonic.

**Problem Addressed**: Random policy never intercepts Mach 5 target.

**Design**:
| Stage        | Speed    | Why This Progression                 |
| ------------ | -------- | ------------------------------------ |
| Basic        | 500 m/s  | Agents can catch up, learn pursuit   |
| Intermediate | 1000 m/s | Introduce need for coordination      |
| Advanced     | 1700 m/s | Full challenge, requires projectiles |

**Rationale**:
- Provides learning signal at each stage
- Skills transfer between stages
- Prevents policy collapse from impossible task
- Measurable progress (advancement criteria)

**Trade-off**: Longer total training, but actually converges.

---

### Decision: Intrinsic Rewards for Exploration

**Choice**: Add velocity mismatch, novelty, and geometry rewards.

**Problem Addressed**: Sparse extrinsic rewards (+100/-100 at episode end) give weak signal.

**Intrinsic Rewards**:
```python
# Velocity mismatch: Don't just follow the target
mismatch = -0.05 * cos(angle(agent_vel, target_vel))

# Approach geometry: Reward intercept course
geometry = +0.1 * cos(angle(approach, optimal_approach))

# Novelty: Visit new states
novelty = +0.01 * state_novelty(current_state)
```

**Rationale**:
- Provides dense learning signal
- Guides exploration toward useful behaviors
- Prevents local minima (e.g., just following target)

**Trade-off**: Risk of reward hacking, but carefully designed rewards align with goal.

---

## 7. Infrastructure Decisions

### Decision: Configuration-Driven Design

**Choice**: All hyperparameters in YAML, not code.

**Alternatives Considered**:
- Hardcoded constants
- Command-line arguments
- Environment variables

**Rationale**:
- Rapid experimentation (no code changes)
- Reproducible experiments (config file is record)
- Easy parameter sweeps
- Clear separation of concerns

**Trade-off**: Extra file to manage, but cleaner codebase.

---

### Decision: Comprehensive Test Suite

**Choice**: 90%+ test coverage with unit and integration tests.

**Alternatives Considered**:
- Minimal testing (rely on manual verification)
- Only integration tests
- Only unit tests

**Rationale**:
- RL bugs are hard to detect (silent failures)
- Physics errors compound over time
- Refactoring requires confidence
- Documentation through tests

**Trade-off**: Development time upfront, but saves debugging time.

---

### Decision: Streamlit Dashboard (Not Custom Web App)

**Choice**: Use Streamlit for interactive visualization.

**Alternatives Considered**:
- Flask/Django custom app
- Jupyter notebooks only
- Desktop GUI (tkinter, PyQt)

**Rationale**:
- Rapid prototyping (Python-native)
- Interactive widgets built-in
- No frontend development needed
- Easy deployment (streamlit share)
- Good for demos and stakeholder presentations

**Trade-off**: Less customizable than custom app, but much faster to build.

---

### Technical Debt Acknowledged

1. **Tight coupling** in some visualization code
2. **Magic numbers** in reward shaping (needs tuning)
3. **Limited type hints** in older modules
4. **Inconsistent logging** across modules

These are tracked for future cleanup.

---

## 9. Decision Summary Table

| Category       | Decision             | Key Rationale                   |
| -------------- | -------------------- | ------------------------------- |
| Environment    | PettingZoo + RLlib   | Standard APIs, ecosystem        |
| Dynamics       | 2D initially         | Fast iteration                  |
| Actions        | Continuous           | Natural for aerospace           |
| Policy         | Shared across agents | Sample efficiency               |
| Specialization | Hierarchical policy  | Role-based coordination         |
| Communication  | GNN                  | Scalable, permutation invariant |
| Projectiles    | PN guidance          | Industry standard               |
| Physics        | ISA atmosphere       | Altitude effects                |
| Integration    | Euler method         | Simplicity                      |
| Sensors        | Probabilistic        | Realistic uncertainty           |
| Fusion         | Kalman filter        | Optimal, interpretable          |
| Training       | Curriculum           | Progressive difficulty          |
| Exploration    | Intrinsic rewards    | Dense signal                    |
| Config         | YAML-driven          | Reproducibility                 |
| Testing        | 90%+ coverage        | Confidence in correctness       |
| Dashboard      | Streamlit            | Rapid prototyping               |
