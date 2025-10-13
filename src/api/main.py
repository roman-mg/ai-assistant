"""FastAPI application with chat endpoints and WebSocket support."""

import asyncio
import json
import time
import uuid

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger

from src.agents.research_agent import research_agent
from src.config.settings import settings
from src.models.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    PaperSearchRequest,
    PaperSearchResponse,
    StreamingChunk,
    WebSocketMessage,
    WebSocketResponse,
)
from src.vectorstore.faiss_store import vector_store

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI/ML Research Chatbot Assistant with LangGraph and FastAPI",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory conversation storage (in production, use a proper database)
conversations: dict[str, list[dict[str, str]]] = {}

# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept a WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific WebSocket connection."""
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        """Broadcast a message to all connected WebSockets."""
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")


manager = ConnectionManager()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        uptime=time.time(),
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint for research queries."""
    try:
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # Get conversation history
        conversation_history = conversations.get(conversation_id, [])
        
        # Add user message to history
        conversation_history.append({
            "role": "user",
            "content": request.message,
        })
        
        logger.info(f"Processing chat request: {request.message}")
        
        # Perform research
        research_result = await research_agent.research(
            query=request.message,
            conversation_history=conversation_history,
        )
        
        # Generate response message
        response_message = await _generate_chat_response(research_result, request.message)
        
        # Add assistant response to history
        conversation_history.append({
            "role": "assistant",
            "content": response_message,
        })
        
        # Update conversation storage
        conversations[conversation_id] = conversation_history[-settings.max_conversation_history:]
        
        # Add papers to vector store for future similarity search
        if research_result.papers:
            research_agent.add_papers_to_vector_store(research_result.papers)
        
        return ChatResponse(
            message=response_message,
            conversation_id=conversation_id,
            research_results=research_result,
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint."""
    try:
        conversation_id = request.conversation_id or str(uuid.uuid4())
        conversation_history = conversations.get(conversation_id, [])
        
        # Add user message to history
        conversation_history.append({
            "role": "user",
            "content": request.message,
        })
        
        async def generate_stream():
            """Generate streaming response."""
            try:
                # Send initial chunk
                yield f"data: {json.dumps(StreamingChunk(content='Starting research...', conversation_id=conversation_id).model_dump())}\n\n"
                
                # Perform research
                research_result = await research_agent.research(
                    query=request.message,
                    conversation_history=conversation_history,
                )
                
                # Stream research results
                if research_result.papers:
                    yield f"data: {json.dumps(StreamingChunk(content=f'Found {len(research_result.papers)} papers', chunk_type='metadata', conversation_id=conversation_id).model_dump())}\n\n"
                    
                    for i, paper in enumerate(research_result.papers):
                        paper_data = {
                            "title": paper.title,
                            "authors": paper.authors,
                            "summary": paper.summary,
                            "url": paper.url,
                        }
                        yield f"data: {json.dumps(StreamingChunk(content=json.dumps(paper_data), chunk_type='paper', conversation_id=conversation_id).model_dump())}\n\n"
                
                # Generate and stream final response
                response_message = await _generate_chat_response(research_result, request.message)
                
                # Stream response in chunks
                words = response_message.split()
                for i, word in enumerate(words):
                    chunk = StreamingChunk(
                        content=word + " ",
                        conversation_id=conversation_id,
                        is_final=(i == len(words) - 1),
                    )
                    yield f"data: {json.dumps(chunk.model_dump())}\n\n"
                    await asyncio.sleep(0.05)  # Small delay for streaming effect
                
                # Update conversation storage
                conversation_history.append({
                    "role": "assistant",
                    "content": response_message,
                })
                conversations[conversation_id] = conversation_history[-settings.max_conversation_history:]
                
                # Add papers to vector store
                if research_result.papers:
                    research_agent.add_papers_to_vector_store(research_result.papers)
                
            except Exception as e:
                logger.error(f"Error in streaming: {e}")
                error_chunk = StreamingChunk(
                    content=f"Error: {str(e)}",
                    conversation_id=conversation_id,
                    is_final=True,
                )
                yield f"data: {json.dumps(error_chunk.model_dump())}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
        
    except Exception as e:
        logger.error(f"Error in streaming chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat."""
    await manager.connect(websocket)
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message_data = json.loads(data)
            ws_message = WebSocketMessage(**message_data)
            
            conversation_id = ws_message.conversation_id or str(uuid.uuid4())
            conversation_history = conversations.get(conversation_id, [])
            
            # Add user message to history
            conversation_history.append({
                "role": "user",
                "content": ws_message.message,
            })
            
            # Send acknowledgment
            await manager.send_personal_message(
                json.dumps(WebSocketResponse(
                    type="ack",
                    content="Processing your request...",
                    conversation_id=conversation_id,
                ).model_dump()),
                websocket,
            )
            
            try:
                # Perform research
                research_result = await research_agent.research(
                    query=ws_message.message,
                    conversation_history=conversation_history,
                )
                
                # Generate response
                response_message = await _generate_chat_response(research_result, ws_message.message)
                
                # Send response
                await manager.send_personal_message(
                    json.dumps(WebSocketResponse(
                        type="response",
                        content=response_message,
                        conversation_id=conversation_id,
                        metadata={
                            "papers_found": len(research_result.papers),
                            "search_time": research_result.search_time,
                        },
                    ).model_dump()),
                    websocket,
                )
                
                # Send papers if any
                if research_result.papers:
                    for paper in research_result.papers:
                        paper_data = {
                            "title": paper.title,
                            "authors": paper.authors,
                            "summary": paper.summary,
                            "url": paper.url,
                        }
                        await manager.send_personal_message(
                            json.dumps(WebSocketResponse(
                                type="paper",
                                content=json.dumps(paper_data),
                                conversation_id=conversation_id,
                            ).model_dump()),
                            websocket,
                        )
                
                # Update conversation storage
                conversation_history.append({
                    "role": "assistant",
                    "content": response_message,
                })
                conversations[conversation_id] = conversation_history[-settings.max_conversation_history:]
                
                # Add papers to vector store
                if research_result.papers:
                    research_agent.add_papers_to_vector_store(research_result.papers)
                
            except Exception as e:
                logger.error(f"Error in WebSocket processing: {e}")
                await manager.send_personal_message(
                    json.dumps(WebSocketResponse(
                        type="error",
                        content=f"Error processing request: {str(e)}",
                        conversation_id=conversation_id,
                    ).model_dump()),
                    websocket,
                )
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@app.post("/papers/search", response_model=PaperSearchResponse)
async def search_papers(request: PaperSearchRequest):
    """Search for similar papers in the vector store."""
    try:
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
        
    except Exception as e:
        logger.error(f"Error searching papers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/papers/count")
async def get_paper_count():
    """Get the total number of papers in the vector store."""
    try:
        count = vector_store.get_paper_count()
        return {"count": count}
        
    except Exception as e:
        logger.error(f"Error getting paper count: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _generate_chat_response(research_result, query: str) -> str:
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
            response_parts.extend([
                f"{i}. **{paper.title}**",
                f"   Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}",
                f"   Summary: {paper.summary[:200]}{'...' if len(paper.summary) > 200 else ''}",
                f"   Link: {paper.url}",
                "",
            ])
        
        if len(research_result.papers) > 5:
            response_parts.append(f"... and {len(research_result.papers) - 5} more papers.")
        
        return "\n".join(response_parts)
        
    except Exception as e:
        logger.error(f"Error generating chat response: {e}")
        return f"I found some papers but encountered an error generating the response: {str(e)}"


@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Server will run on {settings.host}:{settings.port}")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("Shutting down application")
    # Save vector store index
    try:
        vector_store.save_index()
        logger.info("Vector store index saved")
    except Exception as e:
        logger.error(f"Error saving vector store: {e}")
