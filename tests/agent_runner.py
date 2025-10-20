import asyncio
import uuid

from loguru import logger

from src.agents.multi_agent_orchestrator import multi_agent_orchestrator
from src.config.settings import settings
from src.models.schemas import (
    ChatRequest,
    ChatResponse,
    PaperSearchRequest,
    PaperSearchResponse,
    ResearchResult,
)
from src.vectorstore.faiss_store import vector_store

conversations: dict[str, list[dict[str, str]]] = {}


async def chat(request: ChatRequest) -> ChatResponse:
    # Generate conversation ID if not provided
    conversation_id = request.conversation_id or str(uuid.uuid4())

    # Get conversation history
    conversation_history = conversations.get(conversation_id, [])

    # Add user message to history
    conversation_history.append(
        {
            "role": "user",
            "content": request.message,
        }
    )

    logger.info(f"Processing chat request: {request.message}")

    # Perform research
    research_result = await multi_agent_orchestrator.research(
        query=request.message,
        conversation_history=conversation_history,
    )

    # Generate response message
    response_message = await _generate_chat_response(research_result, request.message)

    # Add assistant response to history
    conversation_history.append(
        {
            "role": "assistant",
            "content": response_message,
        }
    )

    # Update conversation storage
    conversations[conversation_id] = conversation_history[-settings.conversation.max_history :]

    # Add papers to vector store for future similarity search
    if research_result.papers:
        multi_agent_orchestrator.add_papers_to_vector_store(research_result.papers)

    return ChatResponse(
        message=response_message,
        conversation_id=conversation_id,
        research_results=research_result,
    )


async def search_papers(request: PaperSearchRequest) -> PaperSearchResponse:
    logger.info(f"Searching papers for: {request.query}")

    # Search vector store
    similar_papers = vector_store.search_similar_papers(
        query=request.query,
        k=request.limit,
        similarity_threshold=request.similarity_threshold,
    )

    # Extract papers and scores
    papers = [paper for paper, score in similar_papers]

    return PaperSearchResponse(
        papers=papers,
        total_found=len(papers),
        search_time=0.0,  # Vector search is very fast
    )


async def get_paper_count() -> dict[str, int]:
    count = vector_store.get_paper_count()
    return {"count": count}


async def _generate_chat_response(research_result: ResearchResult, query: str) -> str:
    """Generate a chat response based on research results."""
    try:
        if not research_result.papers:
            return "I couldn't find any relevant papers for your query. Please try rephrasing your question or being more specific about the research area you're interested in."

        # Create response based on research results
        response_parts = [
            f"I found {len(research_result.papers)} relevant papers for your query about '{query}':",
            "",
        ]

        for i, paper in enumerate(research_result.papers[:5], 1):  # Show top 5 papers
            response_parts.extend(
                [
                    f"{i}. **{paper.title}**",
                    f"   Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}",
                    f"   Summary: {paper.summary[:200]}{'...' if len(paper.summary) > 200 else ''}",
                    f"   Link: {paper.url}",
                    "",
                ]
            )

        if len(research_result.papers) > 5:
            response_parts.append(f"... and {len(research_result.papers) - 5} more papers.")

        return "\n".join(response_parts)

    except Exception as e:
        logger.error(f"Error generating chat response: {e}")
        return f"I found some papers but encountered an error generating the response: {str(e)}"


async def main() -> None:
    result = await get_paper_count()
    logger.info(result)

    request = PaperSearchRequest(query="machine learning", similarity_threshold=0.3)
    result = await search_papers(request)
    logger.info(result)

    request = ChatRequest(message="What are the newest papers on text to speech?")
    result = await chat(request)
    logger.info(result)


if __name__ == "__main__":
    asyncio.run(main())
