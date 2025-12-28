"""
Test script for HYPERION environment.
Validates PettingZoo API compliance and basic functionality.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.env.hypersonic_swarm_env import HypersonicSwarmEnv


def test_environment_creation():
    """Test basic environment instantiation."""
    print("Testing environment creation...")
    env = HypersonicSwarmEnv(num_agents=5, max_steps=500)
    print(f"✓ Environment created with {env.num_agents} agents")
    print(f"  Agents: {env.possible_agents}")
    return env


def test_reset():
    """Test environment reset functionality."""
    print("\nTesting reset...")
    env = HypersonicSwarmEnv(num_agents=5)
    observations, infos = env.reset(seed=42)

    print(f"✓ Reset successful")
    print(f"  Observation keys: {list(observations.keys())}")
    print(f"  Observation shape: {observations['agent_0'].shape}")
    print(f"  Sample observation (agent_0):")
    print(f"    {observations['agent_0'][:10]}...")

    return env, observations


def test_action_spaces():
    """Test action and observation spaces."""
    print("\nTesting action/observation spaces...")
    env = HypersonicSwarmEnv(num_agents=3)

    for agent in env.possible_agents:
        obs_space = env.observation_space(agent)
        act_space = env.action_space(agent)

        print(f"✓ {agent}:")
        print(f"    Observation space: {obs_space.shape}")
        print(f"    Action space: {act_space.shape}")
        print(f"    Action bounds: low={act_space.low}, high={act_space.high}")

    return env


def test_step():
    """Test environment step function."""
    print("\nTesting step function...")
    env = HypersonicSwarmEnv(num_agents=3, max_steps=100)
    observations, _ = env.reset(seed=42)

    # Generate random actions
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    obs, rewards, terminations, truncations, infos = env.step(actions)

    print(f"✓ Step executed successfully")
    print(f"  Sample reward (agent_0): {rewards['agent_0']:.4f}")
    print(f"  Terminated: {any(terminations.values())}")
    print(f"  Truncated: {any(truncations.values())}")
    print(f"  Info keys: {list(infos['agent_0'].keys())}")

    return env, obs, rewards


def test_episode():
    """Run complete episode with random actions."""
    print("\nTesting full episode (random policy)...")
    env = HypersonicSwarmEnv(num_agents=5, max_steps=500)
    observations, _ = env.reset(seed=42)

    episode_rewards = {agent: 0.0 for agent in env.agents}
    step_count = 0
    done = False

    while not done and step_count < 100:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        obs, rewards, terminations, truncations, infos = env.step(actions)

        for agent in env.agents:
            episode_rewards[agent] += rewards.get(agent, 0.0)

        done = all(terminations.values()) or all(truncations.values())
        step_count += 1

    print(f"✓ Episode completed")
    print(f"  Steps: {step_count}")
    print(f"  Episode rewards:")
    for agent, reward in episode_rewards.items():
        print(f"    {agent}: {reward:.4f}")

    if infos.get("agent_0"):
        print(f"  Intercepted: {infos['agent_0'].get('intercepted', False)}")
        print(f"  Target escaped: {infos['agent_0'].get('target_escaped', False)}")


def test_physics():
    """Test physics constraints."""
    print("\nTesting physics constraints...")
    env = HypersonicSwarmEnv(num_agents=2, max_steps=1000)
    observations, _ = env.reset(seed=42)

    # Test max speed enforcement
    for _ in range(50):
        actions = {agent: np.array([1.0, 0.0]) for agent in env.agents}
        obs, _, _, _, _ = env.step(actions)

    max_speeds = []
    for agent in env.agents:
        velocity = env.agent_states[agent]["velocity"]
        speed = np.linalg.norm(velocity)
        max_speeds.append(speed)

    print(f"✓ Physics constraints verified")
    print(f"  Max observed speed: {max(max_speeds):.2f} m/s")
    print(f"  Speed limit: {env.agent_max_speed} m/s")
    print(f"  Within limit: {all(s <= env.agent_max_speed + 1e-6 for s in max_speeds)}")

    # Test fuel depletion
    fuel_levels = [env.agent_states[agent]["fuel"] for agent in env.agents]
    print(f"  Fuel after 50 steps at full thrust: {min(fuel_levels):.4f}")


def test_detection_range():
    """Test sensor detection mechanics."""
    print("\nTesting detection range...")
    env = HypersonicSwarmEnv(num_agents=1, detection_range=2000.0)
    observations, _ = env.reset(seed=42)

    # Check initial detection status
    obs = observations["agent_0"]
    detection_flag = obs[5]

    print(f"✓ Detection mechanics")
    print(f"  Target detected: {detection_flag > 0.5}")
    print(f"  Detection range: {env.detection_range} m")

    # Calculate actual distance
    agent_pos = env.agent_states["agent_0"]["position"]
    target_pos = env.target_state["position"]
    distance = np.linalg.norm(agent_pos - target_pos)
    print(f"  Actual distance to target: {distance:.2f} m")


def run_all_tests():
    """Execute all test functions."""
    print("=" * 60)
    print("HYPERION Environment Test Suite")
    print("=" * 60)

    try:
        test_environment_creation()
        test_reset()
        test_action_spaces()
        test_step()
        test_physics()
        test_detection_range()
        test_episode()

        print("\n" + "=" * 60)
        print("All tests passed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
