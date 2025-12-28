"""Configuration loading utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_env_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract environment configuration."""
    return config.get("environment", {})


def get_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract training configuration."""
    return config.get("training", {})


def get_evaluation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract evaluation configuration."""
    return config.get("evaluation", {})


def get_curriculum_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract curriculum learning configuration."""
    return config.get("curriculum", {})
