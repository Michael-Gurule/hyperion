"""
Visualization utilities for HYPERION environment.
Renders swarm behavior, target trajectories, and sensor coverage.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from typing import Dict, List, Optional, Tuple
import os


class SwarmVisualizer:
    """
    Visualizer for hypersonic swarm environment.
    Creates real-time plots and saved animations.
    """

    def __init__(
        self,
        arena_size: float = 10000.0,
        detection_range: float = 2000.0,
        communication_range: float = 1500.0,
        intercept_range: float = 50.0,
        figsize: Tuple[int, int] = (12, 10),
    ):
        """
        Initialize visualizer.

        Args:
            arena_size: Size of operational area
            detection_range: Sensor detection range
            communication_range: Agent communication range
            intercept_range: Interception range
            figsize: Figure size (width, height)
        """
        self.arena_size = arena_size
        self.detection_range = detection_range
        self.communication_range = communication_range
        self.intercept_range = intercept_range
        self.figsize = figsize

        # Storage for animation
        self.history = {
            "agent_positions": [],
            "agent_velocities": [],
            "target_positions": [],
            "target_velocities": [],
            "intercepted": [],
            "fuel_levels": [],
        }

    def reset_history(self):
        """Clear stored history."""
        for key in self.history:
            self.history[key] = []

    def record_step(self, agent_states: Dict, target_state: Dict, intercepted: bool):
        """
        Record environment state for animation.

        Args:
            agent_states: Dictionary of agent states
            target_state: Target state dictionary
            intercepted: Whether interception occurred
        """
        # Extract agent positions and velocities
        positions = []
        velocities = []
        fuel_levels = []

        for agent_id in sorted(agent_states.keys()):
            state = agent_states[agent_id]
            positions.append(state["position"].copy())
            velocities.append(state["velocity"].copy())
            fuel_levels.append(state["fuel"])

        self.history["agent_positions"].append(positions)
        self.history["agent_velocities"].append(velocities)
        self.history["target_positions"].append(target_state["position"].copy())
        self.history["target_velocities"].append(target_state["velocity"].copy())
        self.history["intercepted"].append(intercepted)
        self.history["fuel_levels"].append(fuel_levels)

    def render_frame(
        self,
        agent_states: Dict,
        target_state: Dict,
        step: int,
        show_detection: bool = True,
        show_communication: bool = True,
    ) -> plt.Figure:
        """
        Render single frame of environment.

        Args:
            agent_states: Dictionary of agent states
            target_state: Target state dictionary
            step: Current timestep
            show_detection: Whether to show detection ranges
            show_communication: Whether to show communication ranges

        Returns:
            Matplotlib figure
        """
        fig, (ax_main, ax_fuel) = plt.subplots(1, 2, figsize=self.figsize)

        # Main tactical view
        ax_main.set_xlim(-self.arena_size * 1.1, self.arena_size * 1.1)
        ax_main.set_ylim(-self.arena_size * 1.1, self.arena_size * 1.1)
        ax_main.set_aspect("equal")
        ax_main.grid(True, alpha=0.3)
        ax_main.set_xlabel("X Position (m)", fontsize=10)
        ax_main.set_ylabel("Y Position (m)", fontsize=10)
        ax_main.set_title(
            f"HYPERION Tactical View - Step {step}", fontsize=12, fontweight="bold"
        )

        # Protected zone (origin)
        protected_zone = plt.Circle(
            (0, 0), 100, color="red", alpha=0.2, label="Protected Zone"
        )
        ax_main.add_patch(protected_zone)

        # Arena boundary
        arena_boundary = plt.Circle(
            (0, 0), self.arena_size, color="gray", fill=False, linestyle="--", alpha=0.5
        )
        ax_main.add_patch(arena_boundary)

        # Plot target
        if target_state["active"]:
            target_pos = target_state["position"]
            target_vel = target_state["velocity"]

            ax_main.scatter(
                target_pos[0],
                target_pos[1],
                c="red",
                s=200,
                marker="*",
                edgecolors="darkred",
                linewidths=2,
                label="Hypersonic Threat",
                zorder=10,
            )

            # Target velocity vector
            ax_main.arrow(
                target_pos[0],
                target_pos[1],
                target_vel[0] * 2,
                target_vel[1] * 2,
                head_width=200,
                head_length=300,
                fc="red",
                ec="darkred",
                alpha=0.6,
                zorder=9,
            )

        # Plot agents
        agent_positions = []
        for i, (agent_id, state) in enumerate(sorted(agent_states.items())):
            if not state["active"]:
                continue

            pos = state["position"]
            vel = state["velocity"]
            fuel = state["fuel"]

            agent_positions.append(pos)

            # Agent color based on fuel level
            color = plt.cm.viridis(fuel)

            # Agent marker
            ax_main.scatter(
                pos[0],
                pos[1],
                c=[color],
                s=150,
                marker="o",
                edgecolors="black",
                linewidths=1.5,
                zorder=8,
                label=f"UAV {i}" if i == 0 else "",
            )

            # Velocity vector
            ax_main.arrow(
                pos[0],
                pos[1],
                vel[0],
                vel[1],
                head_width=100,
                head_length=150,
                fc=color,
                ec="black",
                alpha=0.7,
                zorder=7,
            )

            # Detection range
            if show_detection:
                detection_circle = plt.Circle(
                    pos,
                    self.detection_range,
                    color="blue",
                    fill=False,
                    linestyle=":",
                    alpha=0.3,
                )
                ax_main.add_patch(detection_circle)

            # Communication range
            if show_communication:
                comm_circle = plt.Circle(
                    pos,
                    self.communication_range,
                    color="green",
                    fill=False,
                    linestyle="-.",
                    alpha=0.2,
                )
                ax_main.add_patch(comm_circle)

        # Draw communication links
        if show_communication and len(agent_positions) > 1:
            for i, pos1 in enumerate(agent_positions):
                for j, pos2 in enumerate(agent_positions[i + 1 :], start=i + 1):
                    distance = np.linalg.norm(pos1 - pos2)
                    if distance <= self.communication_range:
                        ax_main.plot(
                            [pos1[0], pos2[0]],
                            [pos1[1], pos2[1]],
                            "g-",
                            alpha=0.2,
                            linewidth=1,
                        )

        ax_main.legend(loc="upper right", fontsize=8)

        # Fuel levels subplot
        agent_ids = sorted(agent_states.keys())
        fuel_levels = [agent_states[aid]["fuel"] for aid in agent_ids]
        colors = [plt.cm.viridis(f) for f in fuel_levels]

        bars = ax_fuel.barh(
            range(len(agent_ids)), fuel_levels, color=colors, edgecolor="black"
        )
        ax_fuel.set_yticks(range(len(agent_ids)))
        ax_fuel.set_yticklabels([f"Agent {i}" for i in range(len(agent_ids))])
        ax_fuel.set_xlabel("Fuel Remaining", fontsize=10)
        ax_fuel.set_title("Agent Fuel Status", fontsize=12, fontweight="bold")
        ax_fuel.set_xlim(0, 1)
        ax_fuel.grid(True, alpha=0.3, axis="x")

        # Add fuel percentage labels
        for i, (bar, fuel) in enumerate(zip(bars, fuel_levels)):
            ax_fuel.text(fuel + 0.02, i, f"{fuel * 100:.1f}%", va="center", fontsize=9)

        plt.tight_layout()
        return fig

    def create_animation(
        self,
        output_path: str = "hyperion_episode.gif",
        fps: int = 10,
        skip_frames: int = 1,
    ):
        """
        Create animation from recorded history.

        Args:
            output_path: Path to save animation
            fps: Frames per second
            skip_frames: Number of frames to skip (for faster animation)
        """
        if len(self.history["agent_positions"]) == 0:
            print("No history recorded. Cannot create animation.")
            return

        print(
            f"Creating animation with {len(self.history['agent_positions'])} frames..."
        )

        fig, ax = plt.subplots(figsize=(10, 10))

        def update(frame_idx):
            ax.clear()

            # Skip frames if requested
            actual_idx = frame_idx * skip_frames
            if actual_idx >= len(self.history["agent_positions"]):
                actual_idx = len(self.history["agent_positions"]) - 1

            # Setup axes
            ax.set_xlim(-self.arena_size * 1.1, self.arena_size * 1.1)
            ax.set_ylim(-self.arena_size * 1.1, self.arena_size * 1.1)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.set_title(
                f"HYPERION - Step {actual_idx}", fontsize=14, fontweight="bold"
            )

            # Protected zone
            protected = plt.Circle((0, 0), 100, color="red", alpha=0.2)
            ax.add_patch(protected)

            # Arena boundary
            boundary = plt.Circle(
                (0, 0),
                self.arena_size,
                color="gray",
                fill=False,
                linestyle="--",
                alpha=0.5,
            )
            ax.add_patch(boundary)

            # Plot target
            target_pos = self.history["target_positions"][actual_idx]
            ax.scatter(
                target_pos[0],
                target_pos[1],
                c="red",
                s=300,
                marker="*",
                edgecolors="darkred",
                linewidths=2,
                label="Threat",
                zorder=10,
            )

            # Plot agents
            agent_positions = self.history["agent_positions"][actual_idx]
            fuel_levels = self.history["fuel_levels"][actual_idx]

            for pos, fuel in zip(agent_positions, fuel_levels):
                color = plt.cm.viridis(fuel)
                ax.scatter(
                    pos[0],
                    pos[1],
                    c=[color],
                    s=150,
                    marker="o",
                    edgecolors="black",
                    linewidths=1.5,
                    zorder=8,
                )

            # Check if intercepted
            if self.history["intercepted"][actual_idx]:
                ax.text(
                    0,
                    self.arena_size * 1.05,
                    "INTERCEPTED!",
                    ha="center",
                    fontsize=20,
                    fontweight="bold",
                    color="green",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

            ax.legend(loc="upper right")

        total_frames = len(self.history["agent_positions"]) // skip_frames
        anim = FuncAnimation(fig, update, frames=total_frames, interval=1000 / fps)

        # Save animation
        anim.save(output_path, writer="pillow", fps=fps)
        print(f"Animation saved to {output_path}")
        plt.close()

    def plot_trajectory_history(self, output_path: str = "hyperion_trajectories.png"):
        """
        Plot full trajectory history of all agents and target.

        Args:
            output_path: Path to save plot
        """
        if len(self.history["agent_positions"]) == 0:
            print("No history recorded. Cannot plot trajectories.")
            return

        fig, ax = plt.subplots(figsize=(12, 12))

        ax.set_xlim(-self.arena_size * 1.1, self.arena_size * 1.1)
        ax.set_ylim(-self.arena_size * 1.1, self.arena_size * 1.1)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X Position (m)", fontsize=12)
        ax.set_ylabel("Y Position (m)", fontsize=12)
        ax.set_title("HYPERION Episode Trajectories", fontsize=14, fontweight="bold")

        # Protected zone
        protected = plt.Circle(
            (0, 0), 100, color="red", alpha=0.2, label="Protected Zone"
        )
        ax.add_patch(protected)

        # Arena boundary
        boundary = plt.Circle(
            (0, 0), self.arena_size, color="gray", fill=False, linestyle="--", alpha=0.5
        )
        ax.add_patch(boundary)

        # Plot target trajectory
        target_trajectory = np.array(self.history["target_positions"])
        ax.plot(
            target_trajectory[:, 0],
            target_trajectory[:, 1],
            "r-",
            linewidth=2,
            alpha=0.7,
            label="Threat Trajectory",
        )
        ax.scatter(
            target_trajectory[0, 0],
            target_trajectory[0, 1],
            c="red",
            s=200,
            marker="s",
            label="Threat Start",
        )
        ax.scatter(
            target_trajectory[-1, 0],
            target_trajectory[-1, 1],
            c="darkred",
            s=200,
            marker="*",
            label="Threat End",
        )

        # Plot agent trajectories
        num_agents = len(self.history["agent_positions"][0])
        colors = plt.cm.tab10(np.linspace(0, 1, num_agents))

        for i in range(num_agents):
            trajectory = []
            for step_positions in self.history["agent_positions"]:
                if i < len(step_positions):
                    trajectory.append(step_positions[i])

            trajectory = np.array(trajectory)
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color=colors[i],
                linewidth=1.5,
                alpha=0.6,
                label=f"Agent {i}",
            )
            ax.scatter(
                trajectory[0, 0],
                trajectory[0, 1],
                c=[colors[i]],
                s=100,
                marker="o",
                edgecolors="black",
            )

        ax.legend(loc="upper right", fontsize=10)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Trajectory plot saved to {output_path}")
        plt.close()
