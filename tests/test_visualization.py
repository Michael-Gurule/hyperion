"""
Test visualization capabilities.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.env.hypersonic_swarm_env import HypersonicSwarmEnv


def test_visualization():
    """Test environment visualization."""
    print("Testing visualization with simple policy...")

    # render_mode set to enable visualizer
    env = HypersonicSwarmEnv(
        num_agents=5,
        max_steps=500,
        render_mode="rgb_array",  # None, or Set to "human" to see live rendering
    )

    observations, _ = env.reset(seed=42)

    episode_rewards = {agent: 0.0 for agent in env.agents}
    step_count = 0
    done = False

    print("Running episode with pursuit policy...")

    while not done and step_count < 200:
        # Simple pursuit policy: head toward target
        actions = {}
        for agent in env.agents:
            obs = observations[agent]

            # Extract target detection (index 5)
            target_detected = obs[5] > 0.5

            if target_detected:
                # Target relative position is at indices 6-7
                target_rel_x = obs[6]
                target_rel_y = obs[7]

                # Simple heading: point toward target
                desired_angle = np.arctan2(target_rel_y, target_rel_x)
                current_heading = env.agent_states[agent]["heading"]

                # Calculate heading change
                angle_diff = desired_angle - current_heading
                angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

                # Normalize to action space [-1, 1]
                heading_change = np.clip(angle_diff / (np.pi / 4), -1.0, 1.0)

                # Full thrust
                actions[agent] = np.array([1.0, heading_change])
            else:
                # No detection: maintain course
                actions[agent] = np.array([0.5, 0.0])

        observations, rewards, terminations, truncations, infos = env.step(actions)

        for agent in env.agents:
            episode_rewards[agent] += rewards.get(agent, 0.0)

        done = all(terminations.values()) or all(truncations.values())
        step_count += 1

        if step_count % 50 == 0:
            print(f"  Step {step_count}")

    print(f"\n✓ Episode completed in {step_count} steps")
    print(f"  Intercepted: {infos.get('agent_0', {}).get('intercepted', False)}")
    print(f"  Target escaped: {infos.get('agent_0', {}).get('target_escaped', False)}")

    # Generate visualizations
    print(f"\nVisualizer status: {env.visualizer}")
    print(
        f"History length: {len(env.visualizer.history['agent_positions']) if env.visualizer else 0}"
    )

    if env.visualizer is not None:
        print("\nGenerating visualizations...")
        env.visualizer.plot_trajectory_history("outputs/test_trajectories.png")
        print("✓ Trajectory plot saved")

        # Uncomment to create animation (requires imagemagick or pillow)
        # env.visualizer.create_animation('outputs/test_animation.gif', fps=10, skip_frames=2)
        # print("✓ Animation saved")

    print("\n" + "=" * 60)
    print("Visualization test completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)
    test_visualization()
