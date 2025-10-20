"""Test script for the 3-agent multi-agent system."""

import asyncio
import sys
import traceback
from pathlib import Path

from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agents.multi_agent_orchestrator import multi_agent_orchestrator


async def test_multi_agent_system() -> None:
    """Test the 3-agent multi-agent system with a simple query."""
    logger.info("Testing 3-Agent Multi-Agent System")
    logger.info("=" * 50)

    try:
        # Test query
        test_query = "machine learning optimization algorithms"
        logger.info(f"Test Query: {test_query}\n")

        # Run research
        logger.info("Running multi-agent research...")
        result = await multi_agent_orchestrator.research(query=test_query, conversation_history=[])

        # Display results
        logger.info("\nResearch Results:")
        logger.info(f"- Total papers found: {result.total_found}")
        logger.info(f"- Search time: {result.search_time:.2f}s")
        logger.info(f"- Sources used: {', '.join(result.sources)}")

        if result.papers:
            logger.info("\nTop 3 Papers:")
            for i, paper in enumerate(result.papers[:3], 1):
                logger.info(f"{i}. {paper.title}")
                logger.info(f"   Authors: {', '.join(paper.authors[:2])}{'...' if len(paper.authors) > 2 else ''}")
                logger.info(f"   Summary: {paper.summary[:100]}{'...' if len(paper.summary) > 100 else ''}\n")

        # Display summary if available
        if hasattr(result, "metadata") and result.metadata and "summary" in result.metadata:
            logger.info("\nComprehensive Summary:")
            logger.info(f"{result.metadata['summary'][:300]}{'...' if len(result.metadata['summary']) > 300 else ''}\n")

        if result.error:
            logger.info(f"Error: {result.error}")

        logger.info("✅ 3-agent multi-agent system test completed successfully!")

    except Exception:
        logger.info(f"❌ Error during test: {traceback.format_exc()}")


if __name__ == "__main__":
    asyncio.run(test_multi_agent_system())
