"""
Experiment tracking for HYPERION training runs.

Supports multiple backends:
- Local JSON logging (always enabled)
- TensorBoard
- Weights & Biases (W&B)
- MLflow
"""

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

from .versioning import RunMetadata, CheckpointMetadata


@dataclass
class TrackerConfig:
    """Configuration for experiment tracking."""

    # Local logging (always on)
    log_dir: str = "./logs"

    # TensorBoard
    use_tensorboard: bool = False
    tensorboard_dir: Optional[str] = None

    # Weights & Biases
    use_wandb: bool = False
    wandb_project: str = "hyperion"
    wandb_entity: Optional[str] = None
    wandb_tags: Optional[List[str]] = None

    # MLflow
    use_mlflow: bool = False
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_name: str = "hyperion"


class TrackerBackend(ABC):
    """Abstract base class for tracking backends."""

    @abstractmethod
    def init_run(self, run_metadata: RunMetadata, config: Dict[str, Any]):
        """Initialize a new run."""
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics at a given step."""
        pass

    @abstractmethod
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        pass

    @abstractmethod
    def log_artifact(self, path: str, name: Optional[str] = None):
        """Log an artifact (file)."""
        pass

    @abstractmethod
    def finish(self, status: str = "completed"):
        """Finish the run."""
        pass


class LocalJSONTracker(TrackerBackend):
    """Local JSON-based experiment tracking."""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir: Optional[Path] = None
        self.metrics_file: Optional[Path] = None
        self.run_metadata: Optional[RunMetadata] = None

    def init_run(self, run_metadata: RunMetadata, config: Dict[str, Any]):
        """Initialize a new run with local JSON logging."""
        self.run_metadata = run_metadata
        self.run_dir = self.log_dir / run_metadata.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Save run config
        config_path = self.run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

        # Save run metadata
        meta_path = self.run_dir / "run_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(run_metadata.to_dict(), f, indent=2, default=str)

        # Initialize metrics file
        self.metrics_file = self.run_dir / "metrics.jsonl"

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to JSONL file."""
        if self.metrics_file is None:
            return

        record = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **metrics,
        }

        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        if self.run_dir is None:
            return

        params_path = self.run_dir / "params.json"
        with open(params_path, "w") as f:
            json.dump(params, f, indent=2, default=str)

    def log_artifact(self, path: str, name: Optional[str] = None):
        """Copy artifact to run directory."""
        if self.run_dir is None:
            return

        import shutil

        src = Path(path)
        if not src.exists():
            return

        dst_name = name or src.name
        dst = self.run_dir / "artifacts" / dst_name
        dst.parent.mkdir(parents=True, exist_ok=True)

        if src.is_file():
            shutil.copy2(src, dst)
        else:
            shutil.copytree(src, dst, dirs_exist_ok=True)

    def finish(self, status: str = "completed"):
        """Finish the run."""
        if self.run_dir is None:
            return

        summary_path = self.run_dir / "summary.json"
        summary = {
            "run_id": self.run_metadata.run_id if self.run_metadata else "",
            "status": status,
            "finished_at": datetime.now().isoformat(),
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)


class TensorBoardTracker(TrackerBackend):
    """TensorBoard experiment tracking."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.writer = None

    def init_run(self, run_metadata: RunMetadata, config: Dict[str, Any]):
        """Initialize TensorBoard writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter

            run_dir = os.path.join(self.log_dir, run_metadata.run_name)
            self.writer = SummaryWriter(log_dir=run_dir)

            # Log config as text
            config_str = json.dumps(config, indent=2, default=str)
            self.writer.add_text("config", f"```json\n{config_str}\n```")

            # Log metadata
            meta_str = json.dumps(run_metadata.to_dict(), indent=2, default=str)
            self.writer.add_text("run_metadata", f"```json\n{meta_str}\n```")

        except ImportError:
            print("Warning: TensorBoard not available. Install with: pip install tensorboard")
            self.writer = None

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to TensorBoard."""
        if self.writer is None:
            return

        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)

    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters to TensorBoard."""
        if self.writer is None:
            return

        # TensorBoard hparams
        hparam_dict = {}
        for key, value in params.items():
            if isinstance(value, (int, float, str, bool)):
                hparam_dict[key] = value

        if hparam_dict:
            self.writer.add_hparams(hparam_dict, {})

    def log_artifact(self, path: str, name: Optional[str] = None):
        """TensorBoard doesn't directly support artifacts."""
        pass

    def finish(self, status: str = "completed"):
        """Close TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()


class WandBTracker(TrackerBackend):
    """Weights & Biases experiment tracking."""

    def __init__(
        self,
        project: str = "hyperion",
        entity: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        self.project = project
        self.entity = entity
        self.tags = tags or []
        self.run = None

    def init_run(self, run_metadata: RunMetadata, config: Dict[str, Any]):
        """Initialize W&B run."""
        try:
            import wandb

            self.run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=run_metadata.run_name,
                id=run_metadata.run_id,
                config=config,
                tags=self.tags + [run_metadata.model_type, run_metadata.algorithm],
                notes=f"Git: {run_metadata.git_commit or 'unknown'}",
                resume="allow",
            )

            # Log additional metadata
            wandb.config.update(
                {
                    "git_commit": run_metadata.git_commit,
                    "git_branch": run_metadata.git_branch,
                    "config_hash": run_metadata.config_hash,
                }
            )

        except ImportError:
            print("Warning: wandb not available. Install with: pip install wandb")
            self.run = None

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to W&B."""
        if self.run is None:
            return

        import wandb

        wandb.log(metrics, step=step)

    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters to W&B."""
        if self.run is None:
            return

        import wandb

        wandb.config.update(params)

    def log_artifact(self, path: str, name: Optional[str] = None):
        """Log artifact to W&B."""
        if self.run is None:
            return

        import wandb

        artifact_name = name or Path(path).stem
        artifact = wandb.Artifact(artifact_name, type="model")
        artifact.add_file(path)
        wandb.log_artifact(artifact)

    def finish(self, status: str = "completed"):
        """Finish W&B run."""
        if self.run is None:
            return

        import wandb

        wandb.finish(exit_code=0 if status == "completed" else 1)


class MLflowTracker(TrackerBackend):
    """MLflow experiment tracking."""

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = "hyperion",
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run = None

    def init_run(self, run_metadata: RunMetadata, config: Dict[str, Any]):
        """Initialize MLflow run."""
        try:
            import mlflow

            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)

            mlflow.set_experiment(self.experiment_name)

            self.run = mlflow.start_run(run_name=run_metadata.run_name)

            # Log metadata as tags
            mlflow.set_tags(
                {
                    "run_id": run_metadata.run_id,
                    "model_type": run_metadata.model_type,
                    "algorithm": run_metadata.algorithm,
                    "git_commit": run_metadata.git_commit or "unknown",
                    "git_branch": run_metadata.git_branch or "unknown",
                    "config_hash": run_metadata.config_hash,
                }
            )

            # Log config as params
            self._log_nested_params(config)

        except ImportError:
            print("Warning: mlflow not available. Install with: pip install mlflow")
            self.run = None

    def _log_nested_params(self, params: Dict[str, Any], prefix: str = ""):
        """Recursively log nested parameters."""
        import mlflow

        for key, value in params.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                self._log_nested_params(value, full_key)
            else:
                # MLflow param values must be strings
                mlflow.log_param(full_key, str(value))

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to MLflow."""
        if self.run is None:
            return

        import mlflow

        mlflow.log_metrics(metrics, step=step)

    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters to MLflow."""
        if self.run is None:
            return

        self._log_nested_params(params)

    def log_artifact(self, path: str, name: Optional[str] = None):
        """Log artifact to MLflow."""
        if self.run is None:
            return

        import mlflow

        mlflow.log_artifact(path)

    def finish(self, status: str = "completed"):
        """Finish MLflow run."""
        if self.run is None:
            return

        import mlflow

        mlflow.set_tag("status", status)
        mlflow.end_run()


class ExperimentTracker:
    """
    Unified experiment tracker that delegates to multiple backends.

    Usage:
        tracker = ExperimentTracker(config)
        tracker.init_run(run_metadata, training_config)
        tracker.log_metrics({"loss": 0.5, "reward": 100}, step=1000)
        tracker.finish()
    """

    def __init__(self, config: Optional[TrackerConfig] = None):
        """
        Initialize experiment tracker.

        Args:
            config: Tracker configuration
        """
        self.config = config or TrackerConfig()
        self.backends: List[TrackerBackend] = []

        # Always use local JSON tracking
        self.backends.append(LocalJSONTracker(self.config.log_dir))

        # Add optional backends
        if self.config.use_tensorboard:
            tb_dir = self.config.tensorboard_dir or os.path.join(self.config.log_dir, "tensorboard")
            self.backends.append(TensorBoardTracker(tb_dir))

        if self.config.use_wandb:
            self.backends.append(
                WandBTracker(
                    project=self.config.wandb_project,
                    entity=self.config.wandb_entity,
                    tags=self.config.wandb_tags,
                )
            )

        if self.config.use_mlflow:
            self.backends.append(
                MLflowTracker(
                    tracking_uri=self.config.mlflow_tracking_uri,
                    experiment_name=self.config.mlflow_experiment_name,
                )
            )

        self.run_metadata: Optional[RunMetadata] = None

    def init_run(self, run_metadata: RunMetadata, config: Dict[str, Any]):
        """
        Initialize a new run across all backends.

        Args:
            run_metadata: Run metadata
            config: Training configuration
        """
        self.run_metadata = run_metadata

        for backend in self.backends:
            try:
                backend.init_run(run_metadata, config)
            except Exception as e:
                print(f"Warning: Failed to init {backend.__class__.__name__}: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """
        Log metrics to all backends.

        Args:
            metrics: Dictionary of metric name -> value
            step: Current step (episode or timestep)
        """
        for backend in self.backends:
            try:
                backend.log_metrics(metrics, step)
            except Exception as e:
                print(f"Warning: Failed to log metrics to {backend.__class__.__name__}: {e}")

    def log_params(self, params: Dict[str, Any]):
        """
        Log hyperparameters to all backends.

        Args:
            params: Dictionary of parameter name -> value
        """
        for backend in self.backends:
            try:
                backend.log_params(params)
            except Exception as e:
                print(f"Warning: Failed to log params to {backend.__class__.__name__}: {e}")

    def log_artifact(self, path: str, name: Optional[str] = None):
        """
        Log an artifact to all backends.

        Args:
            path: Path to artifact file
            name: Optional name for the artifact
        """
        for backend in self.backends:
            try:
                backend.log_artifact(path, name)
            except Exception as e:
                print(f"Warning: Failed to log artifact to {backend.__class__.__name__}: {e}")

    def log_checkpoint(self, checkpoint_metadata: CheckpointMetadata):
        """
        Log checkpoint information.

        Args:
            checkpoint_metadata: Checkpoint metadata
        """
        metrics = {
            f"checkpoint/{k}": v
            for k, v in checkpoint_metadata.metrics.items()
        }
        metrics["checkpoint/episode"] = float(checkpoint_metadata.episode)
        metrics["checkpoint/timestep"] = float(checkpoint_metadata.timestep)

        self.log_metrics(metrics, step=checkpoint_metadata.timestep)

        if checkpoint_metadata.file_path:
            self.log_artifact(checkpoint_metadata.file_path, f"checkpoint_{checkpoint_metadata.checkpoint_type}")

    def finish(self, status: str = "completed"):
        """
        Finish the run across all backends.

        Args:
            status: Final run status
        """
        for backend in self.backends:
            try:
                backend.finish(status)
            except Exception as e:
                print(f"Warning: Failed to finish {backend.__class__.__name__}: {e}")


def create_tracker_from_config(yaml_config: Dict[str, Any]) -> ExperimentTracker:
    """
    Create experiment tracker from YAML config.

    Args:
        yaml_config: Configuration dictionary (from config.yaml)

    Returns:
        Configured ExperimentTracker
    """
    logging_config = yaml_config.get("logging", {})

    tracker_config = TrackerConfig(
        log_dir=logging_config.get("log_dir", "./logs"),
        use_tensorboard=logging_config.get("tensorboard", False),
        use_wandb=logging_config.get("wandb", False),
        wandb_project=logging_config.get("wandb_project", "hyperion"),
        wandb_entity=logging_config.get("wandb_entity"),
        use_mlflow=logging_config.get("mlflow", False),
        mlflow_tracking_uri=logging_config.get("mlflow_tracking_uri"),
        mlflow_experiment_name=logging_config.get("mlflow_experiment_name", "hyperion"),
    )

    return ExperimentTracker(tracker_config)
