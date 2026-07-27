import os
import yaml
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field

def load_config(config_path: str = "config.yaml") -> dict:
    """Read and parse the YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the file is not valid YAML.

    Returns:
        dict: The parsed configuration settings.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if config is None:
                raise ValueError("Configuration file is empty.")
            return config
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML configuration: {str(e)}")

def get_db_path(config: dict) -> str:
    """Extract and resolve the SQLite database path.

    Creates parent directories if they do not exist.

    Args:
        config (dict): The configuration settings.

    Returns:
        str: Absolute or relative path to the SQLite database.
    """
    db_path = config["output"]["sqlite_db"]
    path = Path(db_path)
    if not path.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        path = (project_root / db_path).resolve()
    os.makedirs(path.parent, exist_ok=True)
    return str(path)

def get_csv_path(config: dict) -> str:
    """Extract and resolve the CSV log file path.

    Creates parent directories if they do not exist.

    Args:
        config (dict): The configuration settings.

    Returns:
        str: Absolute or relative path to the CSV log file.
    """
    csv_path = config["output"]["log_csv"]
    path = Path(csv_path)
    if not path.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        path = (project_root / csv_path).resolve()
    os.makedirs(path.parent, exist_ok=True)
    return str(path)

def get_log_path() -> str:
    """Resolve and prepare the logging directory and file path.

    Creates parent directories if they do not exist.

    Returns:
        str: Path to the main application log file.
    """
    log_dir = Path("logs")
    os.makedirs(log_dir, exist_ok=True)
    return str(log_dir / "agent.log")

def validate_config(config: dict) -> None:
    """Validate that config contents conform to schema rules and business validations.

    Args:
        config (dict): Configuration dictionary to validate.

    Raises:
        ValueError: If validation conditions are violated.
    """
    required_keys = ["dataset", "eligibility", "cooldown", "scheduler", "llm", "notification", "output"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required top-level configuration key: '{key}'")

    # LLM Settings Validation
    llm_settings = config.get("llm", {})
    provider = llm_settings.get("provider", "").lower().strip()
    if provider not in ["gemini", "groq", "ollama"]:
        raise ValueError(f"Invalid LLM provider: '{provider}'. Must be one of: gemini, groq, ollama")

    # Cooldown Settings Validation
    cooldown_settings = config.get("cooldown", {})
    days = cooldown_settings.get("days")
    if not isinstance(days, int) or days <= 0:
        raise ValueError(f"Cooldown days must be a positive integer, got: {days}")

    # Scheduler Settings Validation
    scheduler_settings = config.get("scheduler", {})
    interval_hours = scheduler_settings.get("interval_hours")
    if not isinstance(interval_hours, (int, float)) or interval_hours <= 0:
        raise ValueError(f"Scheduler interval_hours must be a positive number, got: {interval_hours}")

def setup_directories(config: dict) -> None:
    """Ensure all paths and directories needed for output files are initialized.

    Args:
        config (dict): The configuration settings.
    """
    logger = logging.getLogger(__name__)
    get_db_path(config)
    get_csv_path(config)
    get_log_path()
    logger.info("All output directories verified")
