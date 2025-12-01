"""
Leibniz Appointment FSM Microservice - FastAPI Application

Provides HTTP REST API for appointment booking with Redis-based session persistence.

Reference:
    leibniz_agent/services/intent/app.py - Service application pattern
    Cloud Transformation document - Phase 6 specifications
"""

import os
import asyncio
import logging
import time
import json
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
import redis.asyncio as redis

from leibniz_agent.services.appointment.config import AppointmentConfig
from leibniz_agent.services.appointment.models import (
    SessionCreateRequest, SessionCreateResponse,
    ProcessInputRequest, ProcessInputResponse,
    SessionStatusResponse, HealthResponse, MetricsResponse,
    AdminClearSessionsResponse
)
from leibniz_agent.services.appointment.fsm_manager import AppointmentFSMManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global state
redis_client: Optional[redis.Redis] = None
config: Optional[AppointmentConfig] = None
app_start_time: float = 0.0

# Session management
SESSION_PREFIX = "appointment:session:"
METRICS_PREFIX = "appointment:metrics:"


# ============================================================================
# Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage service lifecycle (startup/shutdown)
    """
    global redis_client, config, app_start_time

    logger.info(" Starting Leibniz Appointment FSM service...")
    app_start_time = time.time()

    # Load configuration
    try:
        config = AppointmentConfig.from_env()
        logger.info(" Configuration loaded")
    except Exception as e:
        logger.error(f" Failed to load configuration: {e}")
        raise

    # Initialize Redis client
    try:
        redis_client = redis.Redis.from_url(
            config.redis_url,
            decode_responses=True
        )
        # Test connection
        await redis_client.ping()
        logger.info(" Redis connected")
    except Exception as e:
        logger.warning(f"️ Redis connection failed: {e} - service will start but sessions won't persist")
        redis_client = None

    logger.info(" Leibniz Appointment FSM service ready")

    yield

    # Shutdown
    logger.info(" Shutting down Leibniz Appointment FSM service...")

    if redis_client:
        await redis_client.close()
        logger.info(" Redis connection closed")

    logger.info(" Service stopped")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Leibniz Appointment FSM Service",
    description="Stateful appointment booking with Redis persistence",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/api/v1/session/create", response_model=SessionCreateResponse)
async def create_session(request: SessionCreateRequest = None):
    """
    Create a new appointment booking session.

    Generates a unique session ID, initializes FSM, and stores in Redis.

    Returns:
        Session ID and initial response
    """
    if not config:
        raise HTTPException(status_code=503, detail="Service not configured")

    # Generate session ID
    import uuid
    session_id = str(uuid.uuid4())

    # Create FSM instance
    fsm = AppointmentFSMManager(config)

    # Get initial response
    try:
        response = fsm._handle_init()
    except Exception as e:
        logger.error(f" Failed to initialize FSM: {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize session")

    # Store session in Redis
    if redis_client:
        try:
            session_data = fsm.to_dict()
            session_key = f"{SESSION_PREFIX}{session_id}"
            await redis_client.setex(
                session_key,
                config.session_ttl,
                json.dumps(session_data)  # Secure JSON serialization
            )
            logger.info(f" Session created: {session_id}")
        except Exception as e:
            logger.warning(f"️ Failed to store session in Redis: {e}")

    # Update metrics
    await _increment_metric("total_sessions_created")

    return SessionCreateResponse(
        session_id=session_id,
        state=fsm.state.value,
        response=response
    )


@app.post("/api/v1/session/{session_id}/process", response_model=ProcessInputResponse)
async def process_input(session_id: str, request: ProcessInputRequest):
    """
    Process user input for an existing session.

    Retrieves session from Redis, processes input, updates state.

    Returns:
        Processing result with updated state
    """
    if not config:
        raise HTTPException(status_code=503, detail="Service not configured")

    # Retrieve session from Redis
    session_key = f"{SESSION_PREFIX}{session_id}"
    session_data = None

    if redis_client:
        try:
            session_data_str = await redis_client.get(session_key)
            if session_data_str:
                # Parse session data (secure JSON parsing)
                import json
                session_data = json.loads(session_data_str)
            else:
                raise HTTPException(status_code=404, detail="Session not found or expired")
        except Exception as e:
            logger.error(f" Failed to retrieve session {session_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve session")

    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    # Deserialize FSM
    try:
        fsm = AppointmentFSMManager.from_dict(session_data, config)
    except Exception as e:
        logger.error(f" Failed to deserialize session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to load session")

    # Process input
    try:
        result = fsm.process_input(request.user_input)
    except Exception as e:
        logger.error(f" Failed to process input for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to process input")

    # Handle completion/cancellation
    if result["complete"] or result["cancelled"]:
        # Remove session from Redis
        if redis_client:
            try:
                await redis_client.delete(session_key)
                logger.info(f" Session completed/cancelled: {session_id}")
            except Exception as e:
                logger.warning(f"️ Failed to delete completed session {session_id}: {e}")

        # Update completion/cancellation metrics
        if result["complete"]:
            await _increment_metric("completed_sessions_count")
        elif result["cancelled"]:
            await _increment_metric("cancelled_sessions_count")
    else:
        # Update session in Redis with refreshed TTL
        if redis_client:
            try:
                updated_data = fsm.to_dict()
                await redis_client.setex(
                    session_key,
                    config.session_ttl,
                    json.dumps(updated_data)  # Secure JSON serialization
                )
            except Exception as e:
                logger.warning(f"️ Failed to update session {session_id}: {e}")

    # Update request metrics
    await _increment_metric("total_requests")

    return ProcessInputResponse(**result)


@app.get("/api/v1/session/{session_id}/status", response_model=SessionStatusResponse)
async def get_session_status(session_id: str):
    """
    Get status of an existing session.

    Returns current state, collected data, and timestamps.
    """
    if not config:
        raise HTTPException(status_code=503, detail="Service not configured")

    # Retrieve session from Redis
    session_key = f"{SESSION_PREFIX}{session_id}"
    session_data = None

    if redis_client:
        try:
            session_data_str = await redis_client.get(session_key)
            if session_data_str:
                # Parse session data (secure JSON parsing)
                import json
                session_data = json.loads(session_data_str)
            else:
                raise HTTPException(status_code=404, detail="Session not found or expired")
        except Exception as e:
            logger.error(f" Failed to retrieve session {session_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve session")

    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    # Get session metadata
    created_at = "unknown"
    expires_at = "unknown"

    if redis_client:
        try:
            # Get TTL for expiry calculation
            ttl = await redis_client.ttl(session_key)
            if ttl > 0:
                expires_at = datetime.now().timestamp() + ttl
                created_at = expires_at - config.session_ttl
        except Exception as e:
            logger.warning(f"️ Failed to get session TTL for {session_id}: {e}")

    return SessionStatusResponse(
        session_id=session_id,
        state=session_data.get("state", "unknown"),
        data=session_data.get("data", {}),
        created_at=str(created_at),
        updated_at=str(datetime.now()),
        expires_at=str(expires_at)
    )


@app.delete("/api/v1/session/{session_id}")
async def delete_session(session_id: str):
    """
    Delete an existing session.

    Removes session from Redis and returns success.
    """
    if not config:
        raise HTTPException(status_code=503, detail="Service not configured")

    session_key = f"{SESSION_PREFIX}{session_id}"

    if redis_client:
        try:
            deleted = await redis_client.delete(session_key)
            if deleted:
                logger.info(f" Session deleted: {session_id}")
                await _increment_metric("cancelled_sessions_count")
                return {"message": "Session deleted successfully"}
            else:
                raise HTTPException(status_code=404, detail="Session not found")
        except Exception as e:
            logger.error(f" Failed to delete session {session_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to delete session")

    return {"message": "Session deletion not supported (Redis unavailable)"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Checks service status, Redis connectivity, and configuration.
    """
    redis_connected = False
    if redis_client:
        try:
            await asyncio.wait_for(redis_client.ping(), timeout=1.0)
            redis_connected = True
        except:
            redis_connected = False

    config_valid = config is not None

    # Determine overall status
    if not config_valid:
        status = "unhealthy"
    elif not redis_connected:
        status = "degraded"  # Sessions won't persist but service works
    else:
        status = "healthy"

    uptime_seconds = time.time() - app_start_time

    return HealthResponse(
        status=status,
        redis_connected=redis_connected,
        config_valid=config_valid,
        uptime_seconds=uptime_seconds
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get performance metrics.

    Returns session statistics and performance data.
    """
    if not redis_client:
        return MetricsResponse(
            total_sessions_created=0,
            active_sessions_count=0,
            completed_sessions_count=0,
            cancelled_sessions_count=0,
            average_session_duration=0.0
        )

    try:
        # Get metrics from Redis
        total_sessions = int(await redis_client.get(f"{METRICS_PREFIX}total_sessions_created") or 0)
        completed_sessions = int(await redis_client.get(f"{METRICS_PREFIX}completed_sessions_count") or 0)
        cancelled_sessions = int(await redis_client.get(f"{METRICS_PREFIX}cancelled_sessions_count") or 0)

        # Count active sessions
        active_sessions = 0
        cursor = 0
        while True:
            cursor, keys = await redis_client.scan(cursor, match=f"{SESSION_PREFIX}*", count=100)
            active_sessions += len(keys)
            if cursor == 0:
                break

        # Calculate average duration (simplified - would need more complex tracking)
        avg_duration = 0.0  # Placeholder

        return MetricsResponse(
            total_sessions_created=total_sessions,
            active_sessions_count=active_sessions,
            completed_sessions_count=completed_sessions,
            cancelled_sessions_count=cancelled_sessions,
            average_session_duration=avg_duration
        )

    except Exception as e:
        logger.error(f" Failed to retrieve metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")


@app.post("/admin/clear_sessions", response_model=AdminClearSessionsResponse)
async def clear_sessions():
    """
    Clear all appointment sessions (admin endpoint).

    Deletes all session keys from Redis.
    """
    if not redis_client:
        return AdminClearSessionsResponse(sessions_deleted=0, message="Redis not connected")

    try:
        # Find all session keys
        cursor = 0
        keys_deleted = 0

        while True:
            cursor, keys = await redis_client.scan(cursor, match=f"{SESSION_PREFIX}*", count=100)
            if keys:
                deleted = await redis_client.delete(*keys)
                keys_deleted += deleted

            if cursor == 0:
                break

        logger.info(f" Sessions cleared: {keys_deleted} sessions deleted")
        return AdminClearSessionsResponse(
            sessions_deleted=keys_deleted,
            message=f"Successfully deleted {keys_deleted} sessions"
        )

    except Exception as e:
        logger.error(f" Failed to clear sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear sessions: {str(e)}")


# ============================================================================
# Helper Functions
# ============================================================================

async def _increment_metric(metric_name: str):
    """Increment a metric counter in Redis"""
    if redis_client:
        try:
            key = f"{METRICS_PREFIX}{metric_name}"
            await redis_client.incr(key)
        except Exception as e:
            logger.warning(f"️ Failed to increment metric {metric_name}: {e}")


# ============================================================================
# Root Endpoint
# ============================================================================

@app.get("/")
async def root():
    """
    Service information endpoint
    """
    return {
        "service": "Leibniz Appointment FSM Service",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "create_session": "POST /api/v1/session/create",
            "process_input": "POST /api/v1/session/{session_id}/process",
            "get_status": "GET /api/v1/session/{session_id}/status",
            "delete_session": "DELETE /api/v1/session/{session_id}",
            "health": "GET /health",
            "metrics": "GET /metrics",
            "clear_sessions": "POST /admin/clear_sessions"
        }
    }


if __name__ == "__main__":
    # NOTE: This block is for local development only.
    # Docker deployment uses the CMD in Dockerfile which bypasses this.
    import uvicorn
    port = int(os.getenv("PORT", "8005"))
    workers = int(os.getenv("WORKERS", "1"))

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        workers=workers,
        log_level="info"
    )