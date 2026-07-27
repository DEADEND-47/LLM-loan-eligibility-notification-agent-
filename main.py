import os
import time
import logging
import logging.handlers
from contextlib import asynccontextmanager
from typing import Generator

from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler

from agent.scheduler import AgentScheduler
from agent.api import app, init_api
from agent.config import load_config, setup_directories, validate_config, get_log_path

def setup_logging() -> None:
    """Initialize rotating file and console logging configurations."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = get_log_path()

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        # Avoid duplicating handlers if already initialized
        return

    root_logger.setLevel(logging.INFO)
    format_str = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(format_str, datefmt=date_fmt)

    # 5MB Rotating File Handler
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_path,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Console Output Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    logging.info("Logging initialized")

def create_scheduler_app(config_path: str = "config.yaml") -> AgentScheduler:
    """Load configuration files, execute validations, and build the scheduler object.

    Args:
        config_path (str): Filepath to the config settings.

    Returns:
        AgentScheduler: Instantiated agent scheduler orchestrator.
    """
    config = load_config(config_path)
    validate_config(config)
    return AgentScheduler(config_path)

@asynccontextmanager
async def lifespan(app: FastAPI) -> Generator[None, None, None]:
    """FastAPI Lifespan context manager coordinating startup and shutdown routines."""
    setup_logging()
    load_dotenv()
    logging.info("Starting Loan Eligibility Agent")

    # Instantiate scheduler
    agent_scheduler = create_scheduler_app()
    init_api(agent_scheduler)

    # Trigger initial run synchronously on startup to verify state and key validity
    agent_scheduler.run_pipeline()

    # Configure background scheduling thread
    scheduler = BackgroundScheduler(timezone="UTC")
    interval = float(agent_scheduler.scheduler_config.get("interval_hours", 1))
    
    scheduler.add_job(
        func=agent_scheduler.run_pipeline,
        trigger='interval',
        hours=interval
    )
    scheduler.start()
    logging.info(f"Scheduler started with interval of {interval} hours")

    yield

    # Cleanup operations
    scheduler.shutdown(wait=False)
    logging.info("Scheduler stopped")

# Register the lifespan context into the FastAPI instance
app.router.lifespan_context = lifespan

if __name__ == "__main__":
    setup_logging()
    load_dotenv()
    
    uvicorn.run(
        app=app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )
