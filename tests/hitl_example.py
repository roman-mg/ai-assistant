"""Example of using Human-in-the-Loop (HITL) for query confirmation."""

import asyncio
import json

import httpx
from loguru import logger


class HITLExample:
    """Example client for testing HITL functionality."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        """Initialize the HITL example client."""
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

    async def chat_with_hitl(self, query: str, conversation_id: str | None = None) -> dict:
        """
        Send a chat request (HITL is always enabled).

        Args:
            query: Research query
            conversation_id: Optional conversation ID

        Returns:
            Response data
        """
        url = f"{self.base_url}/chat"
        payload = {
            "message": query,
            "conversation_id": conversation_id,
        }

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Sending query (HITL always enabled): {query}")
        logger.info(f"{'=' * 60}\n")

        response = await self.client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

        logger.info("Response:")
        logger.info(json.dumps(data, indent=2))

        return data

    async def get_hitl_session(self, session_id: str) -> dict:
        """
        Get HITL session information.

        Args:
            session_id: Session identifier

        Returns:
            Session data
        """
        url = f"{self.base_url}/hitl/session/{session_id}"

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Getting HITL session: {session_id}")
        logger.info(f"{'=' * 60}\n")

        response = await self.client.get(url)
        response.raise_for_status()
        data = response.json()

        logger.info("Session Info:")
        logger.info(json.dumps(data, indent=2))

        return data

    async def confirm_hitl_query(
        self,
        session_id: str,
        final_query: str,
        user_response: str | None = None,
    ) -> dict:
        """
        Confirm or modify the query and continue with research.

        Args:
            session_id: Session identifier
            final_query: Final confirmed/modified query
            user_response: Optional user feedback

        Returns:
            Confirmation response
        """
        url = f"{self.base_url}/hitl/confirm"
        payload = {
            "session_id": session_id,
            "final_query": final_query,
            "user_response": user_response,
        }

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Confirming query: {final_query}")
        logger.info(f"{'=' * 60}\n")

        response = await self.client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

        logger.info("Confirmation Response:")
        logger.info(json.dumps(data, indent=2))

        return data

    async def reject_hitl_query(self, session_id: str, user_response: str | None = None) -> dict:
        """
        Reject the suggested query improvement.

        Args:
            session_id: Session identifier
            user_response: Optional user feedback

        Returns:
            Rejection response
        """
        url = f"{self.base_url}/hitl/reject/{session_id}"
        params = {}
        if user_response:
            params["user_response"] = user_response

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Rejecting session: {session_id}")
        logger.info(f"{'=' * 60}\n")

        response = await self.client.post(url, params=params)
        response.raise_for_status()
        data = response.json()

        logger.info("Rejection Response:")
        logger.info(json.dumps(data, indent=2))

        return data

    async def list_pending_sessions(self, conversation_id: str | None = None) -> dict:
        """
        List all pending HITL sessions.

        Args:
            conversation_id: Optional conversation ID filter

        Returns:
            List of sessions
        """
        url = f"{self.base_url}/hitl/sessions"
        params = {}
        if conversation_id:
            params["conversation_id"] = conversation_id

        logger.info(f"\n{'=' * 60}")
        logger.info("Listing pending HITL sessions")
        logger.info(f"{'=' * 60}\n")

        response = await self.client.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        logger.info("Pending Sessions:")
        logger.info(json.dumps(data, indent=2))

        return data


async def example_accept_suggestion() -> None:
    """Example: Accept the AI's suggested query improvement."""
    client = HITLExample()

    try:
        logger.info("\n" + "=" * 60)
        logger.info("EXAMPLE 1: Accept AI's Suggested Query")
        logger.info("=" * 60)

        # Step 1: Send query with HITL enabled
        response = await client.chat_with_hitl("Tell me about deep learning")

        # Check if HITL confirmation is required
        metadata = response.get("metadata", {})
        if metadata.get("requires_confirmation"):
            session_id = metadata["hitl_session_id"]

            # Step 2: Get session details
            session = await client.get_hitl_session(session_id)

            # Step 3: Accept the suggested query
            suggested_query = session["suggested_query"]
            await client.confirm_hitl_query(
                session_id=session_id,
                final_query=suggested_query,
                user_response="Looks good, proceeding with suggested query",
            )

        logger.info("\n✓ Example completed successfully!")
    finally:
        await client.close()


async def example_modify_suggestion() -> None:
    """Example: Modify the AI's suggested query."""
    client = HITLExample()

    try:
        logger.info("\n" + "=" * 60)
        logger.info("EXAMPLE 2: Modify AI's Suggested Query")
        logger.info("=" * 60)

        # Step 1: Send query with HITL enabled
        response = await client.chat_with_hitl("machine learning algorithms")

        # Check if HITL confirmation is required
        metadata = response.get("metadata", {})
        if metadata.get("requires_confirmation"):
            session_id = metadata["hitl_session_id"]

            # Step 2: Get session details
            # session = await client.get_hitl_session(session_id)

            # Step 3: Modify and confirm with custom query
            custom_query = "deep learning neural network architectures and optimization techniques"
            await client.confirm_hitl_query(
                session_id=session_id,
                final_query=custom_query,
                user_response="Modified to be more specific about neural networks",
            )

        logger.info("\n✓ Example completed successfully!")
    finally:
        await client.close()


async def example_reject_suggestion() -> None:
    """Example: Reject the AI's suggested query."""
    client = HITLExample()

    try:
        logger.info("\n" + "=" * 60)
        logger.info("EXAMPLE 3: Reject AI's Suggested Query")
        logger.info("=" * 60)

        # Step 1: Send query with HITL enabled
        response = await client.chat_with_hitl("AI research")

        # Check if HITL confirmation is required
        metadata = response.get("metadata", {})
        if metadata.get("requires_confirmation"):
            session_id = metadata["hitl_session_id"]

            # Step 2: Get session details
            await client.get_hitl_session(session_id)

            # Step 3: Reject the suggestion
            await client.reject_hitl_query(
                session_id=session_id,
                user_response="I want to keep my original query as is",
            )

        logger.info("\n✓ Example completed successfully!")
    finally:
        await client.close()


async def example_list_sessions() -> None:
    """Example: List all pending HITL sessions."""
    client = HITLExample()

    try:
        logger.info("\n" + "=" * 60)
        logger.info("EXAMPLE 4: List Pending HITL Sessions")
        logger.info("=" * 60)

        # Create a few sessions
        await client.chat_with_hitl("transformers in NLP")
        await client.chat_with_hitl("computer vision models")

        # List all pending sessions
        await client.list_pending_sessions()

        logger.info("\n✓ Example completed successfully!")

    finally:
        await client.close()


async def main() -> None:
    """Run all examples."""
    logger.info("\n" + "=" * 60)
    logger.info("HITL (Human-in-the-Loop) Examples")
    logger.info("=" * 60)
    logger.info("\nMake sure the API server is running on http://localhost:8000")
    logger.info("before running these examples.\n")

    try:
        # Run examples
        await example_accept_suggestion()
        await asyncio.sleep(1)

        await example_modify_suggestion()
        await asyncio.sleep(1)

        await example_reject_suggestion()
        await asyncio.sleep(1)

        await example_list_sessions()

    except httpx.HTTPError as e:
        logger.info(f"\n✗ HTTP Error: {e}")
        logger.info("Make sure the API server is running!")
    except Exception as e:
        logger.info(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
