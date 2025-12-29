"""
HYPERION Enhanced Dashboard
Comprehensive visualization for scaled swarm training and evaluation.
Supports 50+ agents, curriculum learning, hierarchical policies, and projectile systems.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import yaml
import torch
from pathlib import Path
from datetime import datetime
import sys
import os
from typing import Dict, List, Optional, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Page configuration
st.set_page_config(
    page_title="HYPERION Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants
CURRICULUM_STAGES = {
    0: {"name": "Stage 1: Slow Ballistic", "speed": "1.5x", "evasion": "None"},
    1: {"name": "Stage 2: Medium Weaving", "speed": "2.0x", "evasion": "Basic"},
    2: {"name": "Stage 3: Fast Jinking", "speed": "3.0x", "evasion": "Medium"},
    3: {"name": "Stage 4: Hypersonic Evasive", "speed": "4.0x", "evasion": "Full"},
}


def load_training_history(path: Path) -> Optional[Dict[str, List]]:
    """Load training history from YAML file."""
    if not path.exists():
        return None
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_checkpoint_info(path: Path) -> Optional[Dict]:
    """Load checkpoint and extract metadata."""
    if not path.exists():
        return None
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        info = {
            "file": path.name,
            "size_mb": path.stat().st_size / (1024 * 1024),
            "modified": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        }
        # Extract training info if available
        if isinstance(checkpoint, dict):
            info["keys"] = list(checkpoint.keys())
            if "episode" in checkpoint:
                info["episode"] = checkpoint["episode"]
            if "curriculum_stage" in checkpoint:
                info["curriculum_stage"] = checkpoint["curriculum_stage"]
            if "success_rate" in checkpoint:
                info["success_rate"] = checkpoint["success_rate"]
        return info
    except Exception as e:
        return {"error": str(e)}


def find_checkpoints() -> Dict[str, List[Path]]:
    """Find all checkpoint directories."""
    checkpoints_dir = Path("checkpoints")
    result = {}
    if checkpoints_dir.exists():
        for subdir in checkpoints_dir.iterdir():
            if subdir.is_dir():
                pt_files = list(subdir.glob("*.pt"))
                yaml_files = list(subdir.glob("*.yaml"))
                result[subdir.name] = {
                    "models": pt_files,
                    "history": yaml_files[0] if yaml_files else None
                }
    return result


def main():
    """Main dashboard application."""

    # Header
    st.title("🎯 HYPERION: Enhanced Swarm Intelligence Dashboard")
    st.markdown("*Hypersonic Defense Operations - Training & Evaluation Platform*")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Navigation")
        page = st.radio(
            "Select View",
            [
                "📊 Training Progress",
                "🔬 Checkpoint Analysis",
                "🎮 Live Simulation",
                "📈 Curriculum Metrics",
                "🤖 Role Distribution",
                "ℹ️ System Info",
            ]
        )

        st.markdown("---")
        st.header("📁 Data Source")

        # Find available checkpoints
        checkpoints = find_checkpoints()
        if checkpoints:
            selected_run = st.selectbox(
                "Select Training Run",
                list(checkpoints.keys())
            )
        else:
            selected_run = None
            st.warning("No checkpoints found")

    # Main content based on selected page
    if page == "📊 Training Progress":
        render_training_progress(checkpoints, selected_run)
    elif page == "🔬 Checkpoint Analysis":
        render_checkpoint_analysis(checkpoints, selected_run)
    elif page == "🎮 Live Simulation":
        render_live_simulation()
    elif page == "📈 Curriculum Metrics":
        render_curriculum_metrics(checkpoints, selected_run)
    elif page == "🤖 Role Distribution":
        render_role_distribution(checkpoints, selected_run)
    elif page == "ℹ️ System Info":
        render_system_info()


def render_training_progress(checkpoints: Dict, selected_run: Optional[str]):
    """Render training progress visualization."""
    st.header("📊 Training Progress")

    if not selected_run or not checkpoints.get(selected_run, {}).get("history"):
        st.warning("No training history available. Select a run with training data.")
        return

    history_path = checkpoints[selected_run]["history"]
    history = load_training_history(history_path)

    if not history:
        st.error("Failed to load training history")
        return

    # Summary metrics
    st.subheader("Training Summary")
    col1, col2, col3, col4 = st.columns(4)

    num_episodes = len(history.get("episode_reward", []))
    success_rate = np.mean(history.get("success", [])) * 100 if history.get("success") else 0
    avg_reward = np.mean(history.get("episode_reward", [])) if history.get("episode_reward") else 0
    final_stage = int(history.get("curriculum_stage", [0])[-1]) if history.get("curriculum_stage") else 0

    col1.metric("Total Episodes", num_episodes)
    col2.metric("Success Rate", f"{success_rate:.1f}%")
    col3.metric("Avg Reward", f"{avg_reward:.1f}")
    col4.metric("Current Stage", f"Stage {final_stage + 1}")

    st.markdown("---")

    # Interactive plots with Plotly
    st.subheader("Training Curves")

    # Create episode index
    episodes = list(range(1, num_episodes + 1))

    # Tab layout for different metrics
    tab1, tab2, tab3, tab4 = st.tabs(["Rewards", "Success Rate", "Losses", "Exploration"])

    with tab1:
        # Episode Rewards
        fig = make_subplots(rows=2, cols=1, subplot_titles=("Episode Reward", "Intrinsic Reward"))

        # Episode reward with smoothing
        rewards = history.get("episode_reward", [])
        smoothed_rewards = pd.Series(rewards).rolling(window=10, min_periods=1).mean()

        fig.add_trace(
            go.Scatter(x=episodes, y=rewards, mode='lines', name='Raw', opacity=0.3, line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=episodes, y=smoothed_rewards, mode='lines', name='Smoothed (10)', line=dict(color='blue', width=2)),
            row=1, col=1
        )

        # Intrinsic reward
        intrinsic = history.get("intrinsic_reward", [])
        if intrinsic:
            smoothed_intrinsic = pd.Series(intrinsic).rolling(window=10, min_periods=1).mean()
            fig.add_trace(
                go.Scatter(x=episodes[:len(intrinsic)], y=intrinsic, mode='lines', name='Raw', opacity=0.3, line=dict(color='orange')),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=episodes[:len(intrinsic)], y=smoothed_intrinsic, mode='lines', name='Smoothed', line=dict(color='orange', width=2)),
                row=2, col=1
            )

        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Success rate with curriculum stages
        success = history.get("success", [])
        curriculum = history.get("curriculum_stage", [])

        fig = go.Figure()

        # Success rate (rolling mean)
        window = min(20, len(success) // 5) if len(success) > 5 else 1
        rolling_success = pd.Series(success).rolling(window=window, min_periods=1).mean() * 100

        fig.add_trace(
            go.Scatter(x=episodes[:len(success)], y=rolling_success, mode='lines',
                      name=f'Success Rate (Rolling {window})', fill='tozeroy', line=dict(color='green'))
        )

        # Add curriculum stage transitions as vertical lines
        if curriculum:
            for i in range(1, len(curriculum)):
                if curriculum[i] != curriculum[i-1]:
                    fig.add_vline(x=i, line_dash="dash", line_color="red",
                                 annotation_text=f"Stage {int(curriculum[i]) + 1}")

        fig.update_layout(
            title="Success Rate Over Training",
            xaxis_title="Episode",
            yaxis_title="Success Rate (%)",
            yaxis_range=[0, 100],
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # Stage breakdown
        if curriculum:
            st.subheader("Stage Breakdown")
            stage_data = []
            for stage_id in sorted(set(int(s) for s in curriculum)):
                stage_episodes = [i for i, s in enumerate(curriculum) if int(s) == stage_id]
                stage_success = [success[i] for i in stage_episodes if i < len(success)]
                stage_data.append({
                    "Stage": CURRICULUM_STAGES.get(stage_id, {}).get("name", f"Stage {stage_id + 1}"),
                    "Episodes": len(stage_episodes),
                    "Success Rate": f"{np.mean(stage_success) * 100:.1f}%" if stage_success else "N/A",
                    "Speed": CURRICULUM_STAGES.get(stage_id, {}).get("speed", "?"),
                    "Evasion": CURRICULUM_STAGES.get(stage_id, {}).get("evasion", "?"),
                })
            st.dataframe(pd.DataFrame(stage_data), use_container_width=True, hide_index=True)

    with tab3:
        # Policy and Value losses
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Policy Loss", "Value Loss"))

        policy_loss = history.get("policy_loss", [])
        value_loss = history.get("value_loss", [])

        if policy_loss:
            fig.add_trace(
                go.Scatter(y=policy_loss, mode='lines', name='Policy Loss', line=dict(color='purple')),
                row=1, col=1
            )

        if value_loss:
            fig.add_trace(
                go.Scatter(y=value_loss, mode='lines', name='Value Loss', line=dict(color='red')),
                row=1, col=2
            )

        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        # Exploration metrics
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Policy Entropy", "Role Entropy"))

        entropy = history.get("entropy", [])
        role_entropy = history.get("role_entropy", [])

        if entropy:
            fig.add_trace(
                go.Scatter(y=entropy, mode='lines', name='Policy Entropy', line=dict(color='teal')),
                row=1, col=1
            )

        if role_entropy:
            fig.add_trace(
                go.Scatter(y=role_entropy, mode='lines', name='Role Entropy', line=dict(color='coral')),
                row=1, col=2
            )

        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


def render_checkpoint_analysis(checkpoints: Dict, selected_run: Optional[str]):
    """Render checkpoint analysis page."""
    st.header("🔬 Checkpoint Analysis")

    if not checkpoints:
        st.warning("No checkpoints found")
        return

    # Show all available runs
    st.subheader("Available Training Runs")

    run_info = []
    for run_name, run_data in checkpoints.items():
        models = run_data.get("models", [])
        history = run_data.get("history")

        for model_path in models:
            info = load_checkpoint_info(model_path)
            if info:
                run_info.append({
                    "Run": run_name,
                    "Model": info.get("file", "?"),
                    "Size (MB)": f"{info.get('size_mb', 0):.2f}",
                    "Modified": info.get("modified", "?"),
                    "Episode": info.get("episode", "?"),
                    "Stage": info.get("curriculum_stage", "?"),
                })

    if run_info:
        st.dataframe(pd.DataFrame(run_info), use_container_width=True, hide_index=True)

    # Detailed checkpoint inspection
    st.markdown("---")
    st.subheader("Checkpoint Details")

    if selected_run and checkpoints.get(selected_run, {}).get("models"):
        models = checkpoints[selected_run]["models"]
        selected_model = st.selectbox("Select Checkpoint", models, format_func=lambda x: x.name)

        if selected_model and st.button("Inspect Checkpoint"):
            with st.spinner("Loading checkpoint..."):
                try:
                    checkpoint = torch.load(selected_model, map_location='cpu', weights_only=False)

                    if isinstance(checkpoint, dict):
                        st.success(f"Checkpoint loaded: {len(checkpoint)} top-level keys")

                        # Display structure
                        with st.expander("Checkpoint Structure"):
                            for key in checkpoint.keys():
                                value = checkpoint[key]
                                if isinstance(value, torch.Tensor):
                                    st.write(f"**{key}**: Tensor {list(value.shape)}")
                                elif isinstance(value, dict):
                                    st.write(f"**{key}**: Dict with {len(value)} keys")
                                else:
                                    st.write(f"**{key}**: {type(value).__name__}")

                        # Display stored metrics
                        if "episode" in checkpoint:
                            st.info(f"Episode: {checkpoint['episode']}")
                        if "curriculum_stage" in checkpoint:
                            stage = int(checkpoint['curriculum_stage'])
                            st.info(f"Curriculum Stage: {CURRICULUM_STAGES.get(stage, {}).get('name', f'Stage {stage + 1}')}")
                        if "success_rate" in checkpoint:
                            st.info(f"Success Rate: {checkpoint['success_rate'] * 100:.1f}%")
                    else:
                        st.info(f"Checkpoint is a {type(checkpoint).__name__}")
                except Exception as e:
                    st.error(f"Error loading checkpoint: {e}")


def render_live_simulation():
    """Render live simulation page."""
    st.header("🎮 Live Simulation")

    st.info("Live simulation requires environment initialization. Configure and run below.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Environment Settings")
        num_agents = st.slider("Number of Agents", 10, 100, 50, step=10)
        arena_size = st.slider("Arena Size", 4000, 12000, 8000, step=1000)
        target_speed_mult = st.slider("Target Speed Multiplier", 1.0, 4.0, 1.5, step=0.5)
        use_projectiles = st.checkbox("Enable Projectiles", value=True)

    with col2:
        st.subheader("Visualization Settings")
        max_steps = st.slider("Max Steps", 100, 500, 300, step=50)
        show_trajectories = st.checkbox("Show Trajectories", value=True)
        show_roles = st.checkbox("Show Agent Roles", value=True)
        show_projectiles = st.checkbox("Show Projectiles", value=True)

    if st.button("🚀 Run Simulation", use_container_width=True):
        st.warning("Simulation requires loading trained models. Use `run_evaluation.py` for full evaluation.")

        # Placeholder for simulation
        st.subheader("Simulation Preview (Placeholder)")

        # Generate mock trajectory data
        fig = go.Figure()

        # Mock agent positions
        np.random.seed(42)
        for i in range(min(10, num_agents)):
            t = np.linspace(0, 1, 50)
            x = np.random.randn() * 1000 + t * np.random.randn() * 500
            y = np.random.randn() * 1000 + t * np.random.randn() * 500
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f'Agent {i}',
                                    line=dict(width=2), opacity=0.7))

        # Mock target
        t = np.linspace(0, 1, 50)
        tx = -3000 + t * 6000
        ty = np.sin(t * 4 * np.pi) * 500
        fig.add_trace(go.Scatter(x=tx, y=ty, mode='lines+markers', name='Target',
                                line=dict(color='red', width=3)))

        fig.update_layout(
            title="Simulation Trajectories (Mock Data)",
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            height=500,
            xaxis=dict(scaleanchor="y", scaleratio=1)
        )
        st.plotly_chart(fig, use_container_width=True)


def render_curriculum_metrics(checkpoints: Dict, selected_run: Optional[str]):
    """Render curriculum-specific metrics."""
    st.header("📈 Curriculum Learning Metrics")

    # Curriculum stage overview
    st.subheader("Curriculum Stages")

    stage_df = pd.DataFrame([
        {"Stage": f"Stage {i+1}", **CURRICULUM_STAGES[i]}
        for i in range(4)
    ])
    stage_df["Success Threshold"] = ["80%", "70%", "60%", "50%"]
    stage_df["Min Episodes"] = [100, 150, 200, 300]
    st.dataframe(stage_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    if not selected_run or not checkpoints.get(selected_run, {}).get("history"):
        st.info("Select a training run to see curriculum progression")
        return

    history = load_training_history(checkpoints[selected_run]["history"])
    if not history:
        return

    curriculum = history.get("curriculum_stage", [])
    success = history.get("success", [])
    episode_length = history.get("episode_length", [])

    if not curriculum:
        st.warning("No curriculum data in training history")
        return

    # Stage progression timeline
    st.subheader("Stage Progression Timeline")

    fig = go.Figure()

    episodes = list(range(1, len(curriculum) + 1))
    fig.add_trace(
        go.Scatter(x=episodes, y=curriculum, mode='lines+markers', name='Curriculum Stage',
                  line=dict(color='purple', width=2), marker=dict(size=4))
    )

    fig.update_layout(
        xaxis_title="Episode",
        yaxis_title="Stage",
        yaxis=dict(tickmode='array', tickvals=[0, 1, 2, 3],
                  ticktext=['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']),
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)

    # Per-stage analysis
    st.subheader("Per-Stage Performance")

    stage_metrics = []
    for stage_id in sorted(set(int(s) for s in curriculum)):
        stage_mask = [int(s) == stage_id for s in curriculum]
        stage_success = [success[i] for i, m in enumerate(stage_mask) if m and i < len(success)]
        stage_lengths = [episode_length[i] for i, m in enumerate(stage_mask) if m and i < len(episode_length)]

        stage_metrics.append({
            "Stage": stage_id + 1,
            "Episodes": sum(stage_mask),
            "Successes": sum(stage_success),
            "Success Rate": np.mean(stage_success) * 100 if stage_success else 0,
            "Avg Episode Length": np.mean(stage_lengths) if stage_lengths else 0,
        })

    metrics_df = pd.DataFrame(stage_metrics)

    # Create bar chart
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Success Rate by Stage", "Episode Count by Stage"))

    fig.add_trace(
        go.Bar(x=metrics_df["Stage"], y=metrics_df["Success Rate"], name="Success Rate %",
              marker_color='green'),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(x=metrics_df["Stage"], y=metrics_df["Episodes"], name="Episodes",
              marker_color='blue'),
        row=1, col=2
    )

    fig.update_layout(height=400, showlegend=False)
    fig.update_xaxes(title_text="Stage", row=1, col=1)
    fig.update_xaxes(title_text="Stage", row=1, col=2)
    fig.update_yaxes(title_text="Success Rate (%)", row=1, col=1)
    fig.update_yaxes(title_text="Episode Count", row=1, col=2)

    st.plotly_chart(fig, use_container_width=True)

    # Detailed table
    st.dataframe(metrics_df.style.format({
        "Success Rate": "{:.1f}%",
        "Avg Episode Length": "{:.1f}"
    }), use_container_width=True, hide_index=True)


def render_role_distribution(checkpoints: Dict, selected_run: Optional[str]):
    """Render role distribution analysis."""
    st.header("🤖 Hierarchical Role Distribution")

    # Role overview
    st.subheader("Agent Roles")

    roles = [
        {"Role": "SCOUT", "Icon": "🔍", "Purpose": "Early detection, high speed reconnaissance",
         "Priority": "Maximize detection coverage, stay distant from target"},
        {"Role": "TRACKER", "Icon": "📡", "Purpose": "Maintain target lock, relay tracking info",
         "Priority": "Medium distance, continuous LOS to target"},
        {"Role": "INTERCEPTOR", "Icon": "🎯", "Purpose": "Close in and intercept/fire on target",
         "Priority": "Converge to optimal firing positions"},
        {"Role": "SUPPORT", "Icon": "🛡️", "Purpose": "Backup interceptors, fill coverage gaps",
         "Priority": "Position for contingency intercept"},
    ]

    role_df = pd.DataFrame(roles)
    st.dataframe(role_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Role Assignment Analysis")

    if not selected_run or not checkpoints.get(selected_run, {}).get("history"):
        st.info("Select a training run with hierarchical policy data")

        # Show mock distribution
        fig = go.Figure(data=[
            go.Pie(labels=['SCOUT', 'TRACKER', 'INTERCEPTOR', 'SUPPORT'],
                  values=[15, 10, 20, 5], hole=.4)
        ])
        fig.update_layout(title="Example Role Distribution (Mock Data)")
        st.plotly_chart(fig, use_container_width=True)
        return

    history = load_training_history(checkpoints[selected_run]["history"])
    if not history:
        return

    role_entropy = history.get("role_entropy", [])

    if role_entropy:
        st.subheader("Role Assignment Entropy Over Training")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(y=role_entropy, mode='lines', name='Role Entropy',
                      line=dict(color='coral', width=2))
        )
        fig.add_hline(y=np.log(4), line_dash="dash", line_color="gray",
                     annotation_text="Max Entropy (uniform)")

        fig.update_layout(
            xaxis_title="Training Update",
            yaxis_title="Role Entropy",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info("""
        **Role Entropy Interpretation:**
        - High entropy (~1.39) = Uniform role distribution
        - Low entropy = Specialized role assignments
        - Decreasing entropy over training indicates the policy is learning to specialize roles
        """)


def render_system_info():
    """Render system information page."""
    st.header("ℹ️ System Information")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("HYPERION Architecture")
        st.markdown("""
        ### Core Components

        **Environment (`scaled_environment.py`)**
        - 50-100+ agent support
        - 8000x8000 unit arena
        - Adversarial target with evasion
        - Projectile system with PN guidance

        **Models**
        - `SwarmGNN`: Graph neural network for agent communication
        - `HierarchicalMAPPO`: Multi-agent PPO with role-based policies
        - `ThreatDetector`: Neural threat detection

        **Training Pipeline**
        - Parameterized curriculum learning
        - 4-stage difficulty progression
        - Intrinsic rewards (novelty, geometry, anti-trailing)
        """)

    with col2:
        st.subheader("Curriculum Configuration")

        stage_info = """
        | Stage | Speed | Evasion | Threshold |
        |-------|-------|---------|-----------|
        | 1 | 1.5x (450 m/s) | None | 80% |
        | 2 | 2.0x (600 m/s) | Basic | 70% |
        | 3 | 3.0x (900 m/s) | Medium | 60% |
        | 4 | 4.0x (1200 m/s) | Full | 50% |
        """
        st.markdown(stage_info)

        st.subheader("Reward Structure")
        rewards = """
        - **Intercept**: +100.0
        - **Escape**: -100.0
        - **Projectile Hit**: +50.0
        - **Geometry Bonus**: +1.0
        - **Trailing Penalty**: -0.5
        - **Launch Cost**: -0.5
        """
        st.markdown(rewards)

    st.markdown("---")

    # Check for available data
    st.subheader("Data Status")

    checkpoints = find_checkpoints()
    outputs_dir = Path("outputs")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Training Runs", len(checkpoints))

    with col2:
        total_checkpoints = sum(len(v.get("models", [])) for v in checkpoints.values())
        st.metric("Total Checkpoints", total_checkpoints)

    with col3:
        output_files = list(outputs_dir.glob("*.json")) if outputs_dir.exists() else []
        st.metric("Evaluation Results", len(output_files))

    # Quick links
    st.markdown("---")
    st.subheader("Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📁 Refresh Data", use_container_width=True):
            st.rerun()

    with col2:
        if st.button("📊 Export Summary", use_container_width=True):
            # Create summary JSON
            summary = {
                "timestamp": datetime.now().isoformat(),
                "checkpoints": {k: {"models": [str(p) for p in v.get("models", [])]}
                               for k, v in checkpoints.items()},
            }
            st.json(summary)

    with col3:
        st.button("🔄 Clear Cache", use_container_width=True, on_click=st.cache_data.clear)


if __name__ == "__main__":
    main()
