"""Test script for the 3-agent multi-agent system."""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agents.multi_agent_orchestrator import multi_agent_orchestrator


async def test_multi_agent_system():
    """Test the 3-agent multi-agent system with a simple query."""
    print("Testing 3-Agent Multi-Agent System")
    print("=" * 50)
    
    try:
        # Test query
        test_query = "machine learning optimization algorithms"
        print(f"Test Query: {test_query}")
        print()
        
        # Run research
        print("Running multi-agent research...")
        result = await multi_agent_orchestrator.research(
            query=test_query,
            conversation_history=[]
        )
        
        # Display results
        print(f"\nResearch Results:")
        print(f"- Total papers found: {result.total_found}")
        print(f"- Search time: {result.search_time:.2f}s")
        print(f"- Sources used: {', '.join(result.sources)}")
        
        if result.papers:
            print(f"\nTop 3 Papers:")
            for i, paper in enumerate(result.papers[:3], 1):
                print(f"{i}. {paper.title}")
                print(f"   Authors: {', '.join(paper.authors[:2])}{'...' if len(paper.authors) > 2 else ''}")
                print(f"   Summary: {paper.summary[:100]}{'...' if len(paper.summary) > 100 else ''}")
                print()
        
        # Display summary if available
        if hasattr(result, 'metadata') and result.metadata and 'summary' in result.metadata:
            print(f"\nComprehensive Summary:")
            print(f"{result.metadata['summary'][:300]}{'...' if len(result.metadata['summary']) > 300 else ''}")
            print()
        
        if result.error:
            print(f"Error: {result.error}")
        
        print("✅ 3-agent multi-agent system test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_multi_agent_system())
