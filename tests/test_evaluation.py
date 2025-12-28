"""
Test evaluation metrics.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.env.hypersonic_swarm_env import HypersonicSwarmEnv
from src.evaluation.metrics import EvaluationMetrics, evaluate_policy


def test_metrics_tracking():
    """Test basic metrics tracking."""
    print("Testing metrics tracking...")

    metrics = EvaluationMetrics()

    # Simulate 3 episodes
    for episode in range(3):
        metrics.start_episode()

        # Simulate episode steps
        for step in range(10):
            rewards = {"agent_0": -0.1, "agent_1": -0.1}
            infos = {
                "agent_0": {
                    "intercepted": step == 8 and episode == 0,
                    "target_escaped": step == 9 and episode == 1,
                    "fuel_remaining": 0.8,
                },
                "agent_1": {
                    "intercepted": step == 8 and episode == 0,
                    "target_escaped": step == 9 and episode == 1,
                    "fuel_remaining": 0.7,
                },
            }

            agent_states = {
                "agent_0": {"fuel": 0.8, "position": np.array([100.0, 100.0])},
                "agent_1": {"fuel": 0.7, "position": np.array([120.0, 110.0])},
            }
            target_state = {"position": np.array([150.0, 150.0])}

            metrics.update_step(rewards, infos, agent_states, target_state)

        metrics.end_episode()

    summary = metrics.get_summary_statistics()

    print(f"✓ Metrics tracking test complete")
    print(f"  Episodes: {summary['num_episodes']}")
    print(f"  Interception rate: {summary['interception_rate'] * 100:.1f}%")
    print(f"  Mean episode reward: {summary['mean_episode_reward']:.2f}")

    return summary["num_episodes"] == 3


def test_policy_evaluation():
    """Test full policy evaluation."""
    print("\nTesting policy evaluation...")

    env = HypersonicSwarmEnv(num_agents=3, max_steps=100)

    # Evaluate random policy
    metrics = evaluate_policy(
        env=env,
        policy=None,  # Will use random actions
        num_episodes=10,
        render=False,
    )

    summary = metrics.get_summary_statistics()

    print(f"✓ Policy evaluation test complete")
    print(f"  Episodes evaluated: {summary['num_episodes']}")

    return summary["num_episodes"] == 10


def test_results_saving():
    """Test saving results to file."""
    print("\nTesting results saving...")

    metrics = EvaluationMetrics()

    # Simulate episode
    metrics.start_episode()
    for _ in range(5):
        rewards = {"agent_0": 1.0}
        infos = {"agent_0": {"intercepted": False, "target_escaped": False}}
        metrics.update_step(rewards, infos)
    metrics.end_episode()

    # Save results
    output_path = "outputs/test_eval_results.json"
    metrics.save_results(output_path)

    # Check file exists
    import os

    file_exists = os.path.exists(output_path)

    print(f"✓ Results saving test complete")
    print(f"  File created: {file_exists}")

    return file_exists


def run_all_tests():
    """Execute all evaluation tests."""
    print("=" * 60)
    print("HYPERION Evaluation Metrics Test Suite")
    print("=" * 60)

    results = []

    try:
        results.append(("Metrics Tracking", test_metrics_tracking()))
        results.append(("Policy Evaluation", test_policy_evaluation()))
        results.append(("Results Saving", test_results_saving()))

        print("\n" + "=" * 60)
        print("Test Results Summary")
        print("=" * 60)

        for test_name, passed in results:
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"{test_name}: {status}")

        all_passed = all(result[1] for result in results)

        if all_passed:
            print("\n" + "=" * 60)
            print("All evaluation tests passed!")
            print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)
    run_all_tests()
