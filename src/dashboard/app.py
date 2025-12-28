"""
HYPERION Interactive Dashboard
Streamlit application for visualizing swarm behavior and evaluation metrics.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.env.hypersonic_swarm_env import HypersonicSwarmEnv
from src.evaluation.metrics import evaluate_policy, EvaluationMetrics


# Page configuration
st.set_page_config(
    page_title="HYPERION Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    """Main dashboard application."""

    # Header
    st.title("🚀 HYPERION: Hypersonic Defense Swarm Intelligence")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("Configuration")

        # Environment settings
        st.subheader("Environment")
        num_agents = st.slider("Number of Agents", 1, 10, 5)
        target_speed = st.slider("Target Speed (m/s)", 500, 2000, 1700, step=100)
        max_steps = st.slider("Max Steps", 100, 1000, 500, step=50)

        st.markdown("---")

        # Evaluation settings
        st.subheader("Evaluation")
        num_episodes = st.slider("Episodes to Evaluate", 1, 100, 10)

        st.markdown("---")

        # Action buttons
        run_simulation = st.button("🎮 Run Simulation", use_container_width=True)
        run_evaluation = st.button("📊 Run Evaluation", use_container_width=True)

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🎯 Live Simulation",
            "📈 Performance Metrics",
            "🔍 Episode Analysis",
            "ℹ️ System Info",
        ]
    )

    # Tab 1: Live Simulation
    with tab1:
        st.header("Live Swarm Simulation")

        if run_simulation:
            run_live_simulation(num_agents, target_speed, max_steps)
        else:
            st.info("Click 'Run Simulation' in the sidebar to start")

    # Tab 2: Performance Metrics
    with tab2:
        st.header("Performance Metrics")

        if run_evaluation:
            run_policy_evaluation(num_agents, target_speed, max_steps, num_episodes)
        else:
            st.info("Click 'Run Evaluation' in the sidebar to evaluate policy")

    # Tab 3: Episode Analysis
    with tab3:
        st.header("Episode Analysis")
        display_saved_results()

    # Tab 4: System Info
    with tab4:
        st.header("System Information")
        display_system_info(num_agents, target_speed, max_steps)


def run_live_simulation(num_agents, target_speed, max_steps):
    """Run and visualize a single simulation episode."""

    st.subheader("Simulation in Progress...")

    # Create environment
    env = HypersonicSwarmEnv(
        num_agents=num_agents,
        target_speed=target_speed,
        max_steps=max_steps,
        render_mode="rgb_array",
    )

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Metrics placeholders
    col1, col2, col3, col4 = st.columns(4)
    metric_step = col1.empty()
    metric_reward = col2.empty()
    metric_distance = col3.empty()
    metric_fuel = col4.empty()

    # Visualization placeholder
    viz_placeholder = st.empty()

    # Run episode
    observations, _ = env.reset(seed=42)
    done = False
    step = 0
    total_reward = 0

    trajectory_data = {"steps": [], "rewards": [], "min_distances": [], "avg_fuel": []}

    while not done and step < max_steps:
        # Simple pursuit policy
        actions = {}
        for agent in env.agents:
            obs = observations[agent]
            target_detected = obs[5] > 0.5

            if target_detected:
                target_rel_x = obs[6]
                target_rel_y = obs[7]
                desired_angle = np.arctan2(target_rel_y, target_rel_x)
                current_heading = env.agent_states[agent]["heading"]
                angle_diff = desired_angle - current_heading
                angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
                heading_change = np.clip(angle_diff / (np.pi / 4), -1.0, 1.0)
                actions[agent] = np.array([1.0, heading_change])
            else:
                actions[agent] = np.array([0.5, 0.0])

        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Update metrics
        step += 1
        total_reward += sum(rewards.values())

        # Calculate distance to target
        target_pos = env.target_state["position"]
        min_distance = min(
            [
                np.linalg.norm(state["position"] - target_pos)
                for state in env.agent_states.values()
            ]
        )

        # Calculate average fuel
        avg_fuel = np.mean([state["fuel"] for state in env.agent_states.values()])

        # Store trajectory data
        trajectory_data["steps"].append(step)
        trajectory_data["rewards"].append(total_reward)
        trajectory_data["min_distances"].append(min_distance)
        trajectory_data["avg_fuel"].append(avg_fuel)

        # Update display every 10 steps
        if step % 10 == 0 or step < 10:
            progress_bar.progress(min(step / max_steps, 1.0))
            status_text.text(f"Step {step}/{max_steps}")

            metric_step.metric("Step", step)
            metric_reward.metric("Total Reward", f"{total_reward:.1f}")
            metric_distance.metric("Min Distance", f"{min_distance:.0f}m")
            metric_fuel.metric("Avg Fuel", f"{avg_fuel * 100:.1f}%")

            # Render visualization
            if env.visualizer:
                fig = env.visualizer.render_frame(
                    env.agent_states,
                    env.target_state,
                    step,
                    show_detection=True,
                    show_communication=True,
                )
                viz_placeholder.pyplot(fig)
                plt.close(fig)

        done = all(terminations.values()) or all(truncations.values())

    # Final status
    progress_bar.progress(1.0)

    if infos.get("agent_0", {}).get("intercepted", False):
        st.success(f"✅ Target Intercepted at step {step}!")
    elif infos.get("agent_0", {}).get("target_escaped", False):
        st.error("❌ Target Escaped!")
    else:
        st.warning("⏱️ Episode Timeout")

    # Plot trajectory metrics
    st.subheader("Episode Trajectory")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Reward over time
    axes[0, 0].plot(
        trajectory_data["steps"], trajectory_data["rewards"], "b-", linewidth=2
    )
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Cumulative Reward")
    axes[0, 0].set_title("Reward Over Time")
    axes[0, 0].grid(True, alpha=0.3)

    # Distance to target
    axes[0, 1].plot(
        trajectory_data["steps"], trajectory_data["min_distances"], "r-", linewidth=2
    )
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Distance (m)")
    axes[0, 1].set_title("Minimum Distance to Target")
    axes[0, 1].grid(True, alpha=0.3)

    # Fuel levels
    axes[1, 0].plot(
        trajectory_data["steps"], trajectory_data["avg_fuel"], "g-", linewidth=2
    )
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Fuel Remaining")
    axes[1, 0].set_title("Average Fuel Level")
    axes[1, 0].grid(True, alpha=0.3)

    # Episode summary
    axes[1, 1].axis("off")
    summary_text = f"""
    Episode Summary
    
    Steps: {step}
    Total Reward: {total_reward:.2f}
    Final Distance: {min_distance:.1f}m
    Final Fuel: {avg_fuel * 100:.1f}%
    
    Status: {"Intercepted" if infos.get("agent_0", {}).get("intercepted", False) else "Failed"}
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment="center")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def run_policy_evaluation(num_agents, target_speed, max_steps, num_episodes):
    """Run policy evaluation over multiple episodes."""

    st.subheader(f"Evaluating Random Policy over {num_episodes} Episodes")

    # Create environment
    env = HypersonicSwarmEnv(
        num_agents=num_agents, target_speed=target_speed, max_steps=max_steps
    )

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Run evaluation
    metrics = EvaluationMetrics()

    for episode in range(num_episodes):
        metrics.start_episode()
        observations, _ = env.reset()
        done = False

        while not done:
            # Random policy
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}

            observations, rewards, terminations, truncations, infos = env.step(actions)

            metrics.update_step(
                rewards=rewards,
                infos=infos,
                agent_states=env.agent_states,
                target_state=env.target_state,
            )

            done = all(terminations.values()) or all(truncations.values())

        metrics.end_episode()

        # Update progress
        progress_bar.progress((episode + 1) / num_episodes)
        status_text.text(f"Episode {episode + 1}/{num_episodes}")

    # Get summary statistics
    summary = metrics.get_summary_statistics()

    # Display metrics
    st.success("Evaluation Complete!")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Interception Rate", f"{summary['interception_rate'] * 100:.1f}%")
    col2.metric("Avg Episode Reward", f"{summary['mean_episode_reward']:.1f}")
    col3.metric("Avg Episode Length", f"{summary['mean_episode_length']:.0f}")
    col4.metric("Fuel Efficiency", f"{summary['mean_fuel_efficiency'] * 100:.1f}%")

    # Detailed statistics
    st.subheader("Detailed Statistics")

    stats_df = pd.DataFrame(
        {
            "Metric": [
                "Interception Rate",
                "Escape Rate",
                "Mean Episode Length",
                "Mean Episode Reward",
                "Mean Fuel Efficiency",
                "Mean Min Distance",
            ],
            "Value": [
                f"{summary['interception_rate'] * 100:.1f}%",
                f"{summary['escape_rate'] * 100:.1f}%",
                f"{summary['mean_episode_length']:.1f} steps",
                f"{summary['mean_episode_reward']:.2f}",
                f"{summary['mean_fuel_efficiency'] * 100:.1f}%",
                f"{summary['mean_min_distance']:.1f}m",
            ],
        }
    )

    st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # Save results
    save_path = "outputs/dashboard_evaluation.json"
    metrics.save_results(save_path)
    st.info(f"Results saved to {save_path}")

    # Visualizations
    st.subheader("Performance Distributions")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Episode rewards distribution
    episode_rewards = [ep.total_reward for ep in metrics.episodes]
    axes[0].hist(episode_rewards, bins=20, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Episode Reward")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Episode Reward Distribution")
    axes[0].grid(True, alpha=0.3)

    # Episode length distribution
    episode_lengths = [ep.steps for ep in metrics.episodes]
    axes[1].hist(episode_lengths, bins=20, edgecolor="black", alpha=0.7, color="green")
    axes[1].set_xlabel("Episode Length")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Episode Length Distribution")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def display_saved_results():
    """Display previously saved evaluation results."""

    results_dir = Path("outputs")

    if not results_dir.exists():
        st.warning("No saved results found. Run an evaluation first.")
        return

    # Find JSON files
    json_files = list(results_dir.glob("*.json"))

    if not json_files:
        st.warning("No saved results found. Run an evaluation first.")
        return

    # File selector
    selected_file = st.selectbox(
        "Select Results File", json_files, format_func=lambda x: x.name
    )

    # Load results
    with open(selected_file, "r") as f:
        results = json.load(f)

    summary = results.get("summary", {})
    episodes = results.get("episodes", [])

    # Display summary
    st.subheader("Summary Statistics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Episodes", summary.get("num_episodes", 0))
    col2.metric(
        "Interception Rate", f"{summary.get('interception_rate', 0) * 100:.1f}%"
    )
    col3.metric("Avg Reward", f"{summary.get('mean_episode_reward', 0):.1f}")

    # Episode data table
    st.subheader("Episode Details")

    if episodes:
        episodes_df = pd.DataFrame(episodes)
        st.dataframe(episodes_df, use_container_width=True)

        # Download button
        csv = episodes_df.to_csv(index=False)
        st.download_button(
            label="Download Episode Data (CSV)",
            data=csv,
            file_name=f"{selected_file.stem}_episodes.csv",
            mime="text/csv",
        )


def display_system_info(num_agents, target_speed, max_steps):
    """Display system and configuration information."""

    st.subheader("Environment Configuration")

    config_data = {
        "Parameter": [
            "Number of Agents",
            "Target Speed",
            "Max Steps",
            "Arena Size",
            "Detection Range",
            "Communication Range",
            "Intercept Range",
        ],
        "Value": [
            num_agents,
            f"{target_speed} m/s",
            max_steps,
            "10,000 m",
            "2,000 m",
            "1,500 m",
            "50 m",
        ],
    }

    config_df = pd.DataFrame(config_data)
    st.dataframe(config_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    st.subheader("System Architecture")

    st.markdown("""
    ### HYPERION Components
    
    **Environment**
    - Multi-agent PettingZoo environment
    - Realistic physics simulation
    - Sensor models (RF, thermal)
    - Communication networks
    
    **Detection Module**
    - Multi-sensor fusion (Kalman filtering)
    - Neural network threat detector
    - Multi-target tracking
    
    **Training Pipeline**
    - RLlib with PPO algorithm
    - Curriculum learning
    - Distributed training support
    
    **Evaluation**
    - Comprehensive metrics tracking
    - Performance analysis
    - Results visualization
    """)

    st.markdown("---")

    st.subheader("About HYPERION")

    st.info("""
    HYPERION (Hypersonic Defense Operations) is a sophisticated machine learning 
    platform for simulating and optimizing autonomous drone swarms for hypersonic 
    threat detection and interception.
    
    Built with multi-agent reinforcement learning, advanced sensor fusion, and 
    production-grade evaluation tools.
    """)


if __name__ == "__main__":
    main()
