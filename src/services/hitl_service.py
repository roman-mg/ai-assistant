"""Human-in-the-Loop (HITL) Service for managing query confirmation sessions."""

import time
import uuid
from typing import Any

from loguru import logger


class HITLSession:
    """Represents a HITL confirmation session."""

    def __init__(
        self,
        session_id: str,
        original_query: str,
        suggested_query: str,
        analysis_data: dict[str, Any],
        conversation_id: str | None = None,
    ):
        """Initialize a HITL session."""
        self.session_id = session_id
        self.original_query = original_query
        self.suggested_query = suggested_query
        self.analysis_data = analysis_data
        self.conversation_id = conversation_id
        self.created_at = time.time()
        self.status = "pending"  # pending, confirmed, rejected, modified
        self.final_query: str | None = None
        self.user_response: str | None = None


class HITLService:
    """Service for managing human-in-the-loop query confirmation sessions."""

    def __init__(self):
        """Initialize the HITL service."""
        self.sessions: dict[str, HITLSession] = {}
        self.session_timeout = 3600  # 1 hour timeout for sessions
        logger.info("HITL Service initialized")

    def create_session(
        self,
        original_query: str,
        suggested_query: str,
        analysis_data: dict[str, Any],
        conversation_id: str | None = None,
    ) -> HITLSession:
        """
        Create a new HITL session for query confirmation.

        Args:
            original_query: User's original query
            suggested_query: AI-suggested improved query
            analysis_data: Additional analysis metadata
            conversation_id: Optional conversation context

        Returns:
            Created HITL session
        """
        session_id = str(uuid.uuid4())
        session = HITLSession(
            session_id=session_id,
            original_query=original_query,
            suggested_query=suggested_query,
            analysis_data=analysis_data,
            conversation_id=conversation_id,
        )

        self.sessions[session_id] = session
        logger.info(f"Created HITL session {session_id} for query: {original_query}")

        # Clean up expired sessions
        self._cleanup_expired_sessions()

        return session

    def get_session(self, session_id: str) -> HITLSession | None:
        """
        Get a HITL session by ID.

        Args:
            session_id: Session identifier

        Returns:
            HITL session if found, None otherwise
        """
        session = self.sessions.get(session_id)

        if session and self._is_session_expired(session):
            logger.warning(f"Session {session_id} has expired")
            del self.sessions[session_id]
            return None

        return session

    def confirm_session(
        self,
        session_id: str,
        final_query: str,
        user_response: str | None = None,
    ) -> HITLSession | None:
        """
        Confirm a HITL session with user's final query.

        Args:
            session_id: Session identifier
            final_query: User's final confirmed/modified query
            user_response: Optional user feedback/response

        Returns:
            Updated session if found, None otherwise
        """
        session = self.get_session(session_id)

        if not session:
            logger.warning(f"Attempted to confirm non-existent session {session_id}")
            return None

        session.final_query = final_query
        session.user_response = user_response
        session.status = "confirmed"

        logger.info(f"Session {session_id} confirmed with final query: {final_query}")

        return session

    def reject_session(
        self,
        session_id: str,
        user_response: str | None = None,
    ) -> HITLSession | None:
        """
        Reject a HITL session.

        Args:
            session_id: Session identifier
            user_response: Optional user feedback/response

        Returns:
            Updated session if found, None otherwise
        """
        session = self.get_session(session_id)

        if not session:
            logger.warning(f"Attempted to reject non-existent session {session_id}")
            return None

        session.user_response = user_response
        session.status = "rejected"

        logger.info(f"Session {session_id} rejected by user")

        return session

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a HITL session.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False if not found
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Deleted session {session_id}")
            return True

        return False

    def get_pending_sessions(self, conversation_id: str | None = None) -> list[HITLSession]:
        """
        Get all pending HITL sessions, optionally filtered by conversation.

        Args:
            conversation_id: Optional conversation ID filter

        Returns:
            List of pending sessions
        """
        return [
            session
            for session in self.sessions.values()
            if session.status == "pending" and (conversation_id is None or session.conversation_id == conversation_id)
        ]

    def _is_session_expired(self, session: HITLSession) -> bool:
        """Check if a session has expired."""
        return (time.time() - session.created_at) > self.session_timeout

    def _cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions."""
        current_time = time.time()
        expired_sessions = [
            session_id
            for session_id, session in self.sessions.items()
            if (current_time - session.created_at) > self.session_timeout
        ]

        for session_id in expired_sessions:
            logger.info(f"Cleaning up expired session {session_id}")
            del self.sessions[session_id]


# Global HITL service instance
hitl_service = HITLService()
