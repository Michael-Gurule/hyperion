# Physics and Simulation

HYPERION's effectiveness depends on realistic physics that ensure learned policies can transfer to real-world scenarios. This page outlines the scientific foundations of our simulation.

---

## Why Physics Matter

Training AI agents in simulations with unrealistic physics leads to policies that fail in reality. Consider:

| Unrealistic Assumption | Real-World Failure                     |
| ---------------------- | -------------------------------------- |
| Instant acceleration   | Agents can't execute learned maneuvers |
| No atmospheric effects | Speed/altitude strategies don't work   |
| Perfect sensors        | Agents fail with noisy data            |
| Constant speed targets | Can't handle decelerating threats      |

HYPERION addresses each of these with physically-grounded models.

---

## 1. Atmospheric Model

### International Standard Atmosphere (ISA)

We implement the ISA troposphere model (0-11 km altitude):

**Temperature**:
```
T(h) = T₀ + L·h
```
- T₀ = 288.15 K (sea level)
- L = -0.0065 K/m (lapse rate)

**Pressure** (barometric formula):
```
P(h) = P₀ × (T/T₀)^(-g/(L·R))
```
- P₀ = 101,325 Pa
- g = 9.81 m/s²
- R = 287.05 J/(kg·K)

**Density** (ideal gas law):
```
ρ(h) = P/(R·T)
```

**Speed of Sound**:
```
a(h) = √(γ·R·T)
```
- γ = 1.4 (specific heat ratio for air)

### Altitude Effects

| Altitude | Temperature | Pressure  | Density     | Speed of Sound |
| -------- | ----------- | --------- | ----------- | -------------- |
| 0 km     | 288.1 K     | 101.3 kPa | 1.225 kg/m³ | 340 m/s        |
| 5 km     | 255.7 K     | 54.0 kPa  | 0.736 kg/m³ | 320 m/s        |
| 10 km    | 223.3 K     | 26.5 kPa  | 0.414 kg/m³ | 299 m/s        |

### Implications for RL

1. **Mach number varies**: Same velocity = different Mach at different altitudes
2. **Drag varies**: Lower density at altitude means less drag
3. **Performance varies**: UAV capabilities change with altitude

---

## 2. Aerodynamic Drag

### Drag Equation

```
F_D = ½ρv²C_D A
```

Where:
- ρ = air density (altitude-dependent)
- v = velocity
- C_D = drag coefficient (0.3 typical for streamlined vehicle)
- A = reference area (0.5 m² typical)

### Velocity Decay

A Mach 8 vehicle at sea level experiences significant deceleration:

| Time | Velocity  | Mach | Distance |
| ---- | --------- | ---- | -------- |
| 0s   | 2,720 m/s | 8.0  | 0 km     |
| 20s  | 1,950 m/s | 5.7  | 47 km    |
| 40s  | 1,480 m/s | 4.4  | 81 km    |
| 60s  | 1,170 m/s | 3.4  | 107 km   |

### Training Implications

1. **Time pressure decreases**: Threats slow down, giving more intercept time at range
2. **Altitude strategy**: Flying higher reduces drag (potential evasion tactic)
3. **Optimal intercept point**: Depends on predicted velocity at intercept

---

## 3. UAV Dynamics

### Motion Model

We use a simplified 2D kinematic model:

**Position update**:
```
x(t+dt) = x(t) + vₓ·dt
y(t+dt) = y(t) + vᵧ·dt
```

**Velocity update**:
```
vₓ(t+dt) = vₓ(t) + aₓ·dt
vᵧ(t+dt) = vᵧ(t) + aᵧ·dt
```

**Heading update**:
```
θ(t+dt) = θ(t) + ω·dt
```

### Constraints

| Parameter        | Value      | Rationale            |
| ---------------- | ---------- | -------------------- |
| Max speed        | 300 m/s    | High-performance UAV |
| Max acceleration | 50 m/s²    | ~5g structural limit |
| Max turn rate    | π/4 rad/s  | Aerodynamic limit    |
| Initial fuel     | 1.0        | Normalized capacity  |
| Fuel consumption | 0.001/step | ~100 step endurance  |

### Fuel Dynamics

```python
fuel_consumed = thrust_magnitude * FUEL_CONSUMPTION_RATE * dt
fuel_remaining = max(0, fuel_remaining - fuel_consumed)
```

When fuel reaches zero, thrust is disabled but the agent continues (momentum-only).

---

## 4. Sensor Physics

### Detection Model

Detection probability decreases with range following a quadratic falloff:

```
P_detect = P_base × (1 - d/d_max)²
```

| Distance | Detection Probability |
| -------- | --------------------- |
| 0 km     | 95%                   |
| 10 km    | 42%                   |
| 20 km    | 10%                   |
| 30 km    | 0%                    |

### Measurement Noise

Position measurements have Gaussian noise:

```
measured = true + N(0, σ²)
```

Typical σ values:
- Close range (<5 km): 50 m
- Medium range (5-15 km): 100 m
- Long range (>15 km): 200 m

### Multi-Sensor Fusion

The Kalman filter fuses measurements from multiple sensors:

```
State: x = [x, y, vₓ, vᵧ]ᵀ

Prediction:
x̂ₖ|ₖ₋₁ = F·x̂ₖ₋₁|ₖ₋₁
Pₖ|ₖ₋₁ = F·Pₖ₋₁|ₖ₋₁·Fᵀ + Q

Update:
K = Pₖ|ₖ₋₁·Hᵀ·(H·Pₖ|ₖ₋₁·Hᵀ + R)⁻¹
x̂ₖ|ₖ = x̂ₖ|ₖ₋₁ + K·(zₖ - H·x̂ₖ|ₖ₋₁)
Pₖ|ₖ = (I - K·H)·Pₖ|ₖ₋₁
```

---

## 5. Target Behavior

### Ballistic Trajectory

The baseline target follows a constant-velocity ballistic path:

```
x(t) = x₀ + vₓ·t
y(t) = y₀ + vᵧ·t
```

### Evasive Maneuvers

#### Sinusoidal Weaving

Periodic oscillation perpendicular to flight path:

```
v_evasive = v_base + A·sin(2πft)·n̂_perpendicular
```

- A = amplitude (100 m/s typical)
- f = frequency (0.2 Hz typical)

#### Random Jinking

Sudden, unpredictable direction changes:

```python
if random() < probability:
    θ = uniform(-π/4, π/4)
    v_new = rotate(v_base, θ)
```

### Evasion Levels

| Level  | Weave Amplitude | Jink Probability |
| ------ | --------------- | ---------------- |
| None   | 0               | 0                |
| Basic  | 50 m/s          | 0.01             |
| Medium | 100 m/s         | 0.02             |
| Full   | 200 m/s         | 0.05             |

---

## 6. Projectile Guidance

### Proportional Navigation (PN)

The classic missile guidance law:

```
a_cmd = N × V_c × λ̇
```

Where:
- N = navigation constant (3.0)
- V_c = closing velocity
- λ̇ = line-of-sight rate (rad/s)

### Geometry

```
        Target
          ●─────→ v_target
         /
        / λ (LOS angle)
       /
      ●────→ v_projectile
  Projectile
```

**Line-of-sight angle**:
```
λ = atan2(y_target - y_proj, x_target - x_proj)
```

**LOS rate**:
```
λ̇ = (λ(t) - λ(t-dt)) / dt
```

**Closing velocity**:
```
V_c = -d(distance)/dt
```

### Why PN Works

For a constant-velocity target, PN produces:
1. **Collision course**: Missile converges on intercept point
2. **Minimum miss distance**: Optimal for non-maneuvering targets
3. **Robustness**: Handles moderate target maneuvers

---

## 7. Intercept Calculation

### Predicted Intercept Point

Given target position/velocity and interceptor position/speed:

```python
def calculate_intercept_point(target_pos, target_vel,
                              interceptor_pos, interceptor_speed):
    # Relative geometry
    relative_pos = target_pos - interceptor_pos
    distance = norm(relative_pos)
    target_speed = norm(target_vel)

    # Check feasibility
    if interceptor_speed <= target_speed * 0.9:
        return None  # Cannot intercept

    # Time to intercept (simplified)
    time_to_intercept = distance / (interceptor_speed - target_speed * 0.5)

    # Predicted intercept point
    return target_pos + target_vel * time_to_intercept
```

### Feasibility Conditions

Interception requires:
1. **Speed advantage**: Interceptor faster than target (or favorable geometry)
2. **Fuel sufficiency**: Enough fuel to reach intercept point
3. **Time availability**: Target hasn't escaped before intercept

---

## 8. Simulation Fidelity Trade-offs

### What We Model

| Feature             | Fidelity | Justification                 |
| ------------------- | -------- | ----------------------------- |
| Atmospheric density | High     | Critical for drag/detection   |
| Aerodynamic drag    | High     | Affects intercept timing      |
| Sensor noise        | High     | Essential for robust policies |
| 2D dynamics         | Medium   | Captures key behaviors        |
| UAV constraints     | Medium   | Realistic performance bounds  |

### What We Simplify

| Feature             | Simplification     | Reason                       |
| ------------------- | ------------------ | ---------------------------- |
| 3D dynamics         | 2D only            | Computational efficiency     |
| Weather effects     | Not modeled        | Complexity vs. training time |
| Radar cross-section | Constant detection | Focus on coordination        |
| Structural loads    | Not modeled        | Outside scope                |

### Future Enhancements

1. **3D Environment**: Full spatial dynamics
2. **Weather Effects**: Rain, clouds, wind
3. **RCS Modeling**: Aspect-dependent detection
4. **Terrain**: Ground clutter, line-of-sight blocking

---

## Interactive Exploration

For hands-on exploration of these physics models, see:

**[Physics Walkthrough Notebook](../../notebooks/physics_walkthrough.ipynb)**

This Jupyter notebook includes:
- Atmospheric profile visualization
- Drag effects on trajectories
- Sensor probability curves
- Evasive maneuver comparisons
- Complete simulation integration
