"""
Test training pipeline setup without full training.
Verifies RLlib configuration and environment compatibility.
"""

"""
Test training pipeline setup without full training.
Verifies RLlib configuration and environment compatibility.
"""

import os
import sys
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.env.rllib_wrapper import RLlibHyperionEnv
from src.env.hypersonic_swarm_env import HypersonicSwarmEnv


def env_creator(env_config):
    """Create environment for RLlib."""
    return RLlibHyperionEnv(env_config)


def test_rllib_compatibility():
    """Test that environment works with RLlib."""
    print("Testing RLlib compatibility...")

    # Initialize Ray
    ray.init(ignore_reinit_error=True, num_cpus=2)

    try:
        # Register environment
        register_env("hyperion_test", lambda config: env_creator(config))

        # Environment config
        env_config = {
            "num_agents": 3,
            "max_steps": 100,
            "arena_size": 10000.0,
            "target_speed": 500.0,  # Start with slower target
        }

        # Create dummy environment to get spaces
        dummy_env = env_creator(env_config)
        agent_ids = list(dummy_env.get_agent_ids())
        obs_space = dummy_env.observation_space[
            agent_ids[0]
        ]  # Use dict access instead of method
        act_space = dummy_env.action_space[
            agent_ids[0]
        ]  # Use dict access instead of method

        print(f"✓ Environment created")
        print(f"  Observation space: {obs_space}")
        print(f"  Action space: {act_space}")

        # Configure PPO (minimal config for testing)
        ppo = (
            PPOConfig()
            .environment("hyperion_test", env_config=env_config)
            .framework("torch")
            .resources(num_gpus=0)
            .env_runners(
                num_env_runners=1, num_envs_per_env_runner=1, rollout_fragment_length=50
            )
            .training(
                train_batch_size=200,
                minibatch_size=64,  # Changed from sgd_minibatch_size
                num_epochs=5,  # Changed from num_sgd_iter
            )
            .multi_agent(
                policies={"shared_policy": (None, obs_space, act_space, {})},
                policy_mapping_fn=lambda agent_id,
                episode,
                worker,
                **kwargs: "shared_policy",
            )
        )

        print(f"✓ PPO configuration created")

        # Build algorithm
        algo = ppo.build()
        print(f"✓ Algorithm built successfully")

        # Run single training iteration
        print("\nRunning single training iteration...")
        result = algo.train()

        print(f"✓ Training iteration completed")
        print(f"  Episode reward mean: {result.get('episode_reward_mean', 'N/A')}")
        print(f"  Episode length mean: {result.get('episode_len_mean', 'N/A')}")
        print(f"  Training iteration time: {result.get('time_total_s', 'N/A'):.2f}s")

        # Cleanup
        algo.stop()

        print("\n" + "=" * 60)
        print("RLlib compatibility test PASSED")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ RLlib compatibility test FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        ray.shutdown()


def test_policy_evaluation():
    """Test that trained policy can be evaluated."""
    print("\nTesting policy evaluation...")

    # Create environment
    env = HypersonicSwarmEnv(num_agents=3, max_steps=100)
    observations, _ = env.reset(seed=42)

    # Simulate random policy
    episode_reward = 0
    step_count = 0
    done = False

    while not done and step_count < 100:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        obs, rewards, terminations, truncations, infos = env.step(actions)

        episode_reward += sum(rewards.values())
        done = all(terminations.values()) or all(truncations.values())
        step_count += 1

    print(f"✓ Policy evaluation test complete")
    print(f"  Episode steps: {step_count}")
    print(f"  Total episode reward: {episode_reward:.2f}")
    print(f"  Average reward per step: {episode_reward / step_count:.4f}")

    return True


def run_all_tests():
    """Execute all training setup tests."""
    print("=" * 60)
    print("HYPERION Training Setup Test Suite")
    print("=" * 60)

    results = []

    try:
        results.append(("RLlib Compatibility", test_rllib_compatibility()))
        results.append(("Policy Evaluation", test_policy_evaluation()))

        print("\n" + "=" * 60)
        print("Test Results Summary")
        print("=" * 60)

        for test_name, passed in results:
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"{test_name}: {status}")

        all_passed = all(result[1] for result in results)

        if all_passed:
            print("\nTraining pipeline is ready!")

    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
