"""
Continuous Session Manager for STT/VAD Microservice

Manages persistent VAD sessions with barge-in detection and Redis state persistence.

Reference:
    leibniz_agent/leibniz_continuous_vad.py - Monolith implementation
    leibniz_agent/services/shared/redis_client.py - Redis client
"""

import asyncio
import logging
import time
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass, asdict
import json

from shared.redis_client import get_redis_client

logger = logging.getLogger(__name__)


@dataclass
class ContinuousSessionState:
    """State for a continuous VAD session"""
    session_id: str
    start_time: float
    last_activity: float
    transcripts_count: int = 0
    errors_count: int = 0
    is_agent_speaking: bool = False
    status: str = "active"  # active, paused, error, closed


class ContinuousSessionManager:
    """
    Manages persistent VAD sessions with barge-in detection.

    Ported from leibniz_continuous_vad.py LeibnizContinuousVAD class.
    Adapted for multi-session microservice context.
    """

    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.active_sessions: Dict[str, ContinuousSessionState] = {}
        self.background_tasks: Dict[str, asyncio.Task] = {}
        self.session_locks: Dict[str, asyncio.Lock] = {}

        # Session configuration
        self.session_ttl_seconds = 1800  # 30 minutes
        self.cleanup_interval_seconds = 300  # 5 minutes

        # Start background cleanup task
        self.cleanup_task = asyncio.create_task(self._cleanup_stale_sessions_loop())

    async def create_session(self, session_id: str) -> ContinuousSessionState:
        """
        Create new continuous session.

        Args:
            session_id: Unique session identifier

        Returns:
            ContinuousSessionState: New session state
        """
        async with self._get_session_lock(session_id):
            if session_id in self.active_sessions:
                logger.warning(f"Session {session_id} already exists, returning existing")
                return self.active_sessions[session_id]

            # Create new session state
            session_state = ContinuousSessionState(
                session_id=session_id,
                start_time=time.time(),
                last_activity=time.time()
            )

            self.active_sessions[session_id] = session_state
            self.session_locks[session_id] = asyncio.Lock()

            # Persist to Redis if available
            if self.redis_client:
                await self._save_session_to_redis(session_state)

            logger.info(f" Created continuous session: {session_id}")
            return session_state

    async def get_session(self, session_id: str) -> Optional[ContinuousSessionState]:
        """
        Get active session state.

        Args:
            session_id: Session identifier

        Returns:
            ContinuousSessionState or None if not found
        """
        async with self._get_session_lock(session_id):
            # Check in-memory first
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                # Update activity timestamp
                session.last_activity = time.time()
                if self.redis_client:
                    await self._save_session_to_redis(session)
                return session

            # Try to load from Redis
            if self.redis_client:
                session = await self._load_session_from_redis(session_id)
                if session:
                    self.active_sessions[session_id] = session
                    self.session_locks[session_id] = asyncio.Lock()
                    logger.info(f" Loaded session from Redis: {session_id}")
                    return session

            return None

    async def update_activity(self, session_id: str) -> bool:
        """
        Update session activity timestamp.

        Args:
            session_id: Session identifier

        Returns:
            bool: True if session exists and was updated
        """
        async with self._get_session_lock(session_id):
            session = self.active_sessions.get(session_id)
            if session:
                session.last_activity = time.time()
                if self.redis_client:
                    await self._save_session_to_redis(session)
                return True
            return False

    async def set_agent_speaking(self, session_id: str, is_speaking: bool) -> bool:
        """
        Update agent speaking state for barge-in detection.

        Args:
            session_id: Session identifier
            is_speaking: Whether agent is currently speaking

        Returns:
            bool: True if session exists and was updated
        """
        async with self._get_session_lock(session_id):
            session = self.active_sessions.get(session_id)
            if session:
                session.is_agent_speaking = is_speaking
                if self.redis_client:
                    await self._save_session_to_redis(session)
                logger.debug(f" Session {session_id} agent_speaking: {is_speaking}")
                return True
            return False

    async def close_session(self, session_id: str) -> bool:
        """
        Close continuous session and cleanup resources.

        Args:
            session_id: Session identifier

        Returns:
            bool: True if session was closed
        """
        async with self._get_session_lock(session_id):
            session = self.active_sessions.pop(session_id, None)
            if not session:
                return False

            # Cancel background tasks
            task = self.background_tasks.pop(session_id, None)
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Remove lock
            self.session_locks.pop(session_id, None)

            # Update status and persist
            session.status = "closed"
            if self.redis_client:
                await self._save_session_to_redis(session)

            logger.info(f" Closed continuous session: {session_id}")
            return True

    async def get_session_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all active sessions.

        Returns:
            dict: Session statistics
        """
        stats = {
            "active_sessions": len(self.active_sessions),
            "total_sessions_created": len(self.active_sessions) + len(self.background_tasks),
            "sessions_by_status": {}
        }

        for session in self.active_sessions.values():
            status = session.status
            stats["sessions_by_status"][status] = stats["sessions_by_status"].get(status, 0) + 1

        return stats

    def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create lock for session"""
        if session_id not in self.session_locks:
            self.session_locks[session_id] = asyncio.Lock()
        return self.session_locks[session_id]

    async def _save_session_to_redis(self, session: ContinuousSessionState):
        """Save session state to Redis"""
        if not self.redis_client:
            return

        try:
            key = f"leibniz:stt_vad:session:{session.session_id}"
            data = asdict(session)
            data["serialized_at"] = time.time()

            await self.redis_client.setex(
                key,
                self.session_ttl_seconds,
                json.dumps(data)
            )
        except Exception as e:
            logger.warning(f"Failed to save session {session.session_id} to Redis: {e}")

    async def _load_session_from_redis(self, session_id: str) -> Optional[ContinuousSessionState]:
        """Load session state from Redis"""
        if not self.redis_client:
            return None

        try:
            key = f"leibniz:stt_vad:session:{session_id}"
            data = await self.redis_client.get(key)
            if not data:
                return None

            parsed = json.loads(data)
            session = ContinuousSessionState(**parsed)
            return session
        except Exception as e:
            logger.warning(f"Failed to load session {session_id} from Redis: {e}")
            return None

    async def _cleanup_stale_sessions_loop(self):
        """Background task to cleanup stale sessions"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval_seconds)
                await self._cleanup_stale_sessions()
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")

    async def _cleanup_stale_sessions(self):
        """Remove expired sessions from memory and Redis"""
        current_time = time.time()
        expired_sessions = []

        # Find expired sessions
        for session_id, session in self.active_sessions.items():
            if current_time - session.last_activity > self.session_ttl_seconds:
                expired_sessions.append(session_id)

        # Close expired sessions
        for session_id in expired_sessions:
            logger.info(f" Cleaning up stale session: {session_id}")
            await self.close_session(session_id)

        if expired_sessions:
            logger.info(f" Cleaned up {len(expired_sessions)} stale sessions")

    async def shutdown(self):
        """Shutdown manager and cleanup all resources"""
        logger.info(" Shutting down ContinuousSessionManager...")

        # Cancel cleanup task
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

        # Close all active sessions
        session_ids = list(self.active_sessions.keys())
        for session_id in session_ids:
            await self.close_session(session_id)

        logger.info(" ContinuousSessionManager shutdown complete")