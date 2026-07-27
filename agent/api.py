import os
import logging
import asyncio
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from agent.scheduler import AgentScheduler
from agent.llm.factory import get_llm_health

scheduler_instance: Optional[AgentScheduler] = None
logger = logging.getLogger(__name__)

def init_api(scheduler: AgentScheduler) -> None:
    """Register the main active scheduler orchestrator instance to the API module context.

    Args:
        scheduler (AgentScheduler): The active pipeline scheduler.
    """
    global scheduler_instance
    scheduler_instance = scheduler
    logger.info("API initialized with scheduler instance")

app = FastAPI(
    title="Loan Eligibility Notification Agent",
    description="Automated loan notification pipeline API",
    version="1.0.0"
)

@app.get("/")
async def root() -> dict:
    """Root metadata API endpoint.

    Returns:
        dict: Basic info, version, and routing references.
    """
    return {
        "message": "Loan Eligibility Notification Agent",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health() -> JSONResponse:
    """Liveness, scheduler, and LLM API connectivity health check endpoint.

    Returns:
        JSONResponse: Provider state report.
    """
    if scheduler_instance is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "message": "Scheduler not initialized"
            }
        )

    llm_report = get_llm_health(scheduler_instance.llm)
    now_iso = datetime.now(timezone.utc).isoformat()

    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "timestamp": now_iso,
            "llm": llm_report,
            "scheduler": "running"
        }
    )

@app.post("/pipeline/run")
async def trigger_pipeline() -> dict:
    """Manually trigger the notification batch runner in a separate thread context.

    Raises:
        HTTPException: If the scheduler context has not been registered.

    Returns:
        dict: Completion summary details.
    """
    if scheduler_instance is None:
        raise HTTPException(status_code=503, detail="Scheduler not initialized")

    # Run the CPU-bound pipeline logic inside an asynchronous executor thread
    result = await asyncio.to_thread(scheduler_instance.run_pipeline)
    now_iso = datetime.now(timezone.utc).isoformat()

    return {
        "triggered": True,
        "timestamp": now_iso,
        "result": result
    }

@app.get("/stats")
async def stats() -> dict:
    """Read summary totals from the persistent SQLite notification history logs.

    Raises:
        HTTPException: If the scheduler context has not been registered.

    Returns:
        dict: Database tracking summary counts.
    """
    if scheduler_instance is None:
        raise HTTPException(status_code=503, detail="Scheduler not initialized")

    db_stats = scheduler_instance.dedup.get_stats()
    now_iso = datetime.now(timezone.utc).isoformat()

    return {
        "timestamp": now_iso,
        "stats": db_stats
    }

@app.get("/logs")
async def get_logs(lines: int = 50) -> dict:
    """Access the final log entries written to the rotation file agent.log.

    Args:
        lines (int): Number of trailing log statements to read.

    Raises:
        HTTPException: For count requests out of bounds (1 - 1000).

    Returns:
        dict: Array of read text records.
    """
    if lines < 1 or lines > 1000:
        raise HTTPException(status_code=400, detail="lines must be between 1 and 1000")

    log_filepath = "logs/agent.log"
    if not os.path.exists(log_filepath):
        return {
            "log_file": log_filepath,
            "lines_requested": lines,
            "lines_returned": 0,
            "logs": []
        }

    try:
        with open(log_filepath, "r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()
        
        selected_lines = [line.rstrip("\r\n") for line in all_lines[-lines:]]
        return {
            "log_file": log_filepath,
            "lines_requested": lines,
            "lines_returned": len(selected_lines),
            "logs": selected_lines
        }
    except Exception as e:
        logger.error(f"Failed reading application log file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Could not read log file: {str(e)}")
