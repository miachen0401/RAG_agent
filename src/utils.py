"""
Utility modules for configuration and logging.

This module provides:
- Configuration loading from YAML
- Logging setup and management
"""

import os
import logging
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv


def load_env_file(env_file: str = ".env"):
    """
    Load environment variables from .env file.

    Args:
        env_file: Path to .env file (default: .env)
    """
    if os.path.exists(env_file):
        load_dotenv(env_file)
        return True
    return False


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Automatically loads .env file first if it exists.
    Supports environment variable substitution in format: ${VAR_NAME}

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    # Load .env file first
    load_env_file()

    with open(config_path, 'r') as f:
        config_str = f.read()

    # Replace environment variables
    for key, value in os.environ.items():
        config_str = config_str.replace(f"${{{key}}}", value)

    config = yaml.safe_load(config_str)
    return config


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        config: Configuration dictionary with logging settings

    Returns:
        Configured logger
    """
    log_config = config.get("logging", {})
    log_level = log_config.get("level", "INFO")
    log_format = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_file = log_config.get("file", "logs/rag_system.log")
    console = log_config.get("console", True)

    # Create logs directory if needed
    log_dir = Path(log_file).parent
    log_dir.mkdir(exist_ok=True, parents=True)

    # Configure root logger
    logger = logging.getLogger("rag_system")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers = []

    formatter = logging.Formatter(log_format)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler (optional)
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Suppress HTTP client logs (set to DEBUG level)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(f"rag_system.{name}")


def load_prompts(prompts_path: str = "prompts/prompts.yaml") -> Dict[str, Any]:
    """
    Load prompts configuration from YAML file.

    DEPRECATED: Use load_llm_config() instead.

    Args:
        prompts_path: Path to prompts configuration file

    Returns:
        Prompts configuration dictionary
    """
    with open(prompts_path, 'r', encoding='utf-8') as f:
        prompts = yaml.safe_load(f)
    return prompts


def load_llm_config(config_name: str, config_dir: str = "llm_configs") -> Dict[str, Any]:
    """
    Load LLM configuration from YAML file.

    Args:
        config_name: Name of config file (e.g., "router_config.yaml" or "router_config")
        config_dir: Directory containing LLM configs (default: "llm_configs")

    Returns:
        LLM configuration dictionary

    Example:
        router_config = load_llm_config("router_config.yaml")
        rag_config = load_llm_config("rag_config")  # .yaml extension optional
    """
    # Add .yaml extension if not present
    if not config_name.endswith('.yaml'):
        config_name = f"{config_name}.yaml"

    config_path = Path(config_dir) / config_name

    if not config_path.exists():
        raise FileNotFoundError(f"LLM config not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config
