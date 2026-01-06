"""
Model training versioning and experiment tracking for HYPERION.

Provides:
- Unique run identification (UUID + timestamp)
- Git commit tracking
- Configuration hashing and versioning
- Checkpoint metadata management
- Model registry for tracking experiments
"""

import os
import json
import hashlib
import subprocess
import uuid
from datetime import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml


@dataclass
class RunMetadata:
    """Metadata for a training run."""

    # Unique identifiers
    run_id: str
    run_name: str
    timestamp: str

    # Code versioning
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    git_dirty: bool = False

    # Configuration
    config_hash: str = ""
    config: Dict[str, Any] = field(default_factory=dict)

    # Training info
    model_type: str = ""  # "hierarchical", "mappo", "ppo"
    algorithm: str = ""

    # Hardware
    device: str = ""

    # Status
    status: str = "running"  # "running", "completed", "failed", "interrupted"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunMetadata":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class CheckpointMetadata:
    """Metadata for a model checkpoint."""

    # Run reference
    run_id: str
    run_name: str

    # Checkpoint info
    checkpoint_type: str  # "episode", "best", "final"
    episode: int = 0
    timestep: int = 0

    # Timestamps
    created_at: str = ""

    # Code versioning
    git_commit: Optional[str] = None

    # Configuration hash
    config_hash: str = ""

    # Performance metrics at checkpoint time
    metrics: Dict[str, float] = field(default_factory=dict)

    # File info
    file_path: str = ""
    file_size_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointMetadata":
        """Create from dictionary."""
        return cls(**data)


def get_git_info() -> Dict[str, Any]:
    """
    Get current git repository information.

    Returns:
        Dict with commit hash, branch name, and dirty status
    """
    git_info = {
        "commit": None,
        "branch": None,
        "dirty": False,
    }

    try:
        # Get commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_info["commit"] = result.stdout.strip()[:12]  # Short hash

        # Get branch name
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_info["branch"] = result.stdout.strip()

        # Check if working directory is dirty
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_info["dirty"] = len(result.stdout.strip()) > 0

    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        # Git not available or not in a repository
        pass

    return git_info


def hash_config(config: Union[Dict[str, Any], Any]) -> str:
    """
    Create a deterministic hash of configuration.

    Args:
        config: Configuration dict or dataclass

    Returns:
        SHA256 hash of the configuration (first 12 chars)
    """
    if hasattr(config, "__dict__"):
        # Dataclass or object
        config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith("_")}
    else:
        config_dict = config

    # Sort keys for deterministic output
    config_str = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:12]


def generate_run_id() -> str:
    """Generate a unique run ID."""
    return str(uuid.uuid4())[:8]


def generate_run_name(prefix: str = "run") -> str:
    """
    Generate a human-readable run name with timestamp.

    Args:
        prefix: Prefix for the run name

    Returns:
        Run name like "run_20240115_143022"
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def create_run_metadata(
    config: Union[Dict[str, Any], Any],
    run_name: Optional[str] = None,
    model_type: str = "hierarchical",
    algorithm: str = "MAPPO",
    device: str = "cpu",
) -> RunMetadata:
    """
    Create metadata for a new training run.

    Args:
        config: Training configuration
        run_name: Optional custom run name
        model_type: Type of model being trained
        algorithm: Training algorithm
        device: Compute device

    Returns:
        RunMetadata instance
    """
    git_info = get_git_info()

    if hasattr(config, "__dict__"):
        config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith("_")}
    else:
        config_dict = config

    return RunMetadata(
        run_id=generate_run_id(),
        run_name=run_name or generate_run_name(model_type),
        timestamp=datetime.now().isoformat(),
        git_commit=git_info["commit"],
        git_branch=git_info["branch"],
        git_dirty=git_info["dirty"],
        config_hash=hash_config(config),
        config=config_dict,
        model_type=model_type,
        algorithm=algorithm,
        device=device,
        status="running",
    )


def create_checkpoint_metadata(
    run_metadata: RunMetadata,
    checkpoint_type: str,
    episode: int = 0,
    timestep: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    file_path: str = "",
) -> CheckpointMetadata:
    """
    Create metadata for a checkpoint.

    Args:
        run_metadata: Parent run metadata
        checkpoint_type: Type of checkpoint ("episode", "best", "final")
        episode: Episode number
        timestep: Total timesteps
        metrics: Performance metrics at checkpoint time
        file_path: Path to checkpoint file

    Returns:
        CheckpointMetadata instance
    """
    file_size = 0
    if file_path and os.path.exists(file_path):
        file_size = os.path.getsize(file_path)

    return CheckpointMetadata(
        run_id=run_metadata.run_id,
        run_name=run_metadata.run_name,
        checkpoint_type=checkpoint_type,
        episode=episode,
        timestep=timestep,
        created_at=datetime.now().isoformat(),
        git_commit=run_metadata.git_commit,
        config_hash=run_metadata.config_hash,
        metrics=metrics or {},
        file_path=file_path,
        file_size_bytes=file_size,
    )


class ModelRegistry:
    """
    Registry for tracking model checkpoints and training runs.

    Maintains a JSON database of all runs and checkpoints for easy
    querying and comparison.
    """

    def __init__(self, registry_path: str = "./checkpoints/registry.json"):
        """
        Initialize registry.

        Args:
            registry_path: Path to registry JSON file
        """
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self):
        """Load registry from disk."""
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                data = json.load(f)
                self.runs = {r["run_id"]: RunMetadata.from_dict(r) for r in data.get("runs", [])}
                self.checkpoints = [CheckpointMetadata.from_dict(c) for c in data.get("checkpoints", [])]
        else:
            self.runs: Dict[str, RunMetadata] = {}
            self.checkpoints: List[CheckpointMetadata] = []

    def _save(self):
        """Save registry to disk."""
        data = {
            "runs": [r.to_dict() for r in self.runs.values()],
            "checkpoints": [c.to_dict() for c in self.checkpoints],
        }
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def register_run(self, metadata: RunMetadata):
        """Register a new training run."""
        self.runs[metadata.run_id] = metadata
        self._save()

    def update_run_status(self, run_id: str, status: str):
        """Update run status."""
        if run_id in self.runs:
            self.runs[run_id].status = status
            self._save()

    def register_checkpoint(self, metadata: CheckpointMetadata):
        """Register a checkpoint."""
        self.checkpoints.append(metadata)
        self._save()

    def get_run(self, run_id: str) -> Optional[RunMetadata]:
        """Get run by ID."""
        return self.runs.get(run_id)

    def get_runs_by_status(self, status: str) -> List[RunMetadata]:
        """Get all runs with given status."""
        return [r for r in self.runs.values() if r.status == status]

    def get_checkpoints_for_run(self, run_id: str) -> List[CheckpointMetadata]:
        """Get all checkpoints for a run."""
        return [c for c in self.checkpoints if c.run_id == run_id]

    def get_best_checkpoint(
        self,
        metric: str = "success_rate",
        model_type: Optional[str] = None,
    ) -> Optional[CheckpointMetadata]:
        """
        Get best checkpoint across all runs.

        Args:
            metric: Metric to optimize
            model_type: Filter by model type

        Returns:
            Best checkpoint or None
        """
        candidates = self.checkpoints

        if model_type:
            run_ids = {r.run_id for r in self.runs.values() if r.model_type == model_type}
            candidates = [c for c in candidates if c.run_id in run_ids]

        # Filter to checkpoints with the metric
        candidates = [c for c in candidates if metric in c.metrics]

        if not candidates:
            return None

        return max(candidates, key=lambda c: c.metrics.get(metric, 0))

    def list_runs(
        self,
        model_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 10,
    ) -> List[RunMetadata]:
        """
        List runs with optional filtering.

        Args:
            model_type: Filter by model type
            status: Filter by status
            limit: Maximum number of runs to return

        Returns:
            List of run metadata
        """
        runs = list(self.runs.values())

        if model_type:
            runs = [r for r in runs if r.model_type == model_type]
        if status:
            runs = [r for r in runs if r.status == status]

        # Sort by timestamp (newest first)
        runs.sort(key=lambda r: r.timestamp, reverse=True)

        return runs[:limit]

    def compare_runs(
        self,
        run_ids: List[str],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple runs.

        Args:
            run_ids: List of run IDs to compare
            metrics: Metrics to compare (uses best checkpoint metrics)

        Returns:
            Comparison dict
        """
        comparison = {}

        for run_id in run_ids:
            run = self.runs.get(run_id)
            if not run:
                continue

            # Get best checkpoint for this run
            run_checkpoints = self.get_checkpoints_for_run(run_id)
            best_ckpt = None
            if run_checkpoints:
                # Prefer "best" type, otherwise use final
                best_candidates = [c for c in run_checkpoints if c.checkpoint_type == "best"]
                if best_candidates:
                    best_ckpt = best_candidates[-1]
                else:
                    best_ckpt = run_checkpoints[-1]

            comparison[run_id] = {
                "run_name": run.run_name,
                "model_type": run.model_type,
                "status": run.status,
                "timestamp": run.timestamp,
                "config_hash": run.config_hash,
                "git_commit": run.git_commit,
                "metrics": best_ckpt.metrics if best_ckpt else {},
            }

        return comparison


def save_run_metadata(metadata: RunMetadata, checkpoint_dir: str):
    """
    Save run metadata alongside checkpoints.

    Args:
        metadata: Run metadata
        checkpoint_dir: Directory containing checkpoints
    """
    path = Path(checkpoint_dir) / "run_metadata.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(metadata.to_dict(), f, default_flow_style=False)


def load_run_metadata(checkpoint_dir: str) -> Optional[RunMetadata]:
    """
    Load run metadata from checkpoint directory.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        RunMetadata or None
    """
    path = Path(checkpoint_dir) / "run_metadata.yaml"

    if not path.exists():
        return None

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return RunMetadata.from_dict(data)


def save_checkpoint_metadata(metadata: CheckpointMetadata, checkpoint_path: str):
    """
    Save checkpoint metadata alongside the checkpoint file.

    Args:
        metadata: Checkpoint metadata
        checkpoint_path: Path to checkpoint .pt file
    """
    meta_path = Path(checkpoint_path).with_suffix(".meta.yaml")

    with open(meta_path, "w") as f:
        yaml.dump(metadata.to_dict(), f, default_flow_style=False)


def load_checkpoint_metadata(checkpoint_path: str) -> Optional[CheckpointMetadata]:
    """
    Load checkpoint metadata.

    Args:
        checkpoint_path: Path to checkpoint .pt file

    Returns:
        CheckpointMetadata or None
    """
    meta_path = Path(checkpoint_path).with_suffix(".meta.yaml")

    if not meta_path.exists():
        return None

    with open(meta_path, "r") as f:
        data = yaml.safe_load(f)

    return CheckpointMetadata.from_dict(data)
