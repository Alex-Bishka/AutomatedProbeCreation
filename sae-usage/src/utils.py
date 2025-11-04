"""Utility functions for configuration loading and common helpers."""

import os
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_env_vars() -> Dict[str, str]:
    """Load environment variables from .env file.

    Returns:
        Dictionary containing API keys and other env vars
    """
    load_dotenv()

    return {
        'goodfire_api_key': os.getenv('GOODFIRE_API_KEY'),
        'openrouter_api_key': os.getenv('OPENROUTER_API_KEY') or os.getenv('OPEN_ROUTER_API_KEY'),
        'hf_token': os.getenv('HF_TOKEN'),
    }


def create_output_dir(base_dir: str, timestamp_format: str = "%Y%m%d_%H%M%S") -> Path:
    """Create timestamped output directory.

    Args:
        base_dir: Base output directory
        timestamp_format: Format for timestamp in directory name

    Returns:
        Path to created output directory
    """
    timestamp = datetime.now().strftime(timestamp_format)
    output_dir = Path(base_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_yaml(data: Dict[str, Any], filepath: Path) -> None:
    """Save data to YAML file.

    Args:
        data: Data to save
        filepath: Path to save file
    """
    with open(filepath, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, indent=2)


def get_model_layer(model_name: str, custom_layer: int = None) -> int:
    """Get appropriate layer for a given model.

    Args:
        model_name: Name of the model
        custom_layer: Custom layer to use (overrides defaults)

    Returns:
        Layer number to extract activations from
    """
    if custom_layer is not None:
        return custom_layer

    # Default layers based on Goodfire SAE availability
    model_layers = {
        "meta-llama/Llama-3.1-8B-Instruct": 19,
        "meta-llama/Llama-3.3-70B-Instruct": 50,
    }

    return model_layers.get(model_name, 19)  # Default to 19 if unknown
