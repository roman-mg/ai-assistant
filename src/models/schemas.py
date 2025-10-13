"""Pydantic models for API requests and responses."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Individual chat message."""

    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)


class ChatRequest(BaseModel):
    """Chat request model."""

    message: str = Field(..., description="User message", min_length=1)
    conversation_id: str | None = Field(
        default=None,
        description="Conversation ID for maintaining context",
    )
    stream: bool = Field(default=False, description="Enable streaming response")
    max_papers: int | None = Field(
        default=None,
        description="Maximum number of papers to return",
        ge=1,
        le=20,
    )


class Paper(BaseModel):
    """Research paper model."""

    title: str = Field(..., description="Paper title")
    authors: list[str] = Field(..., description="list of authors")
    abstract: str = Field(..., description="Paper abstract")
    summary: str = Field(..., description="AI-generated summary")
    url: str = Field(..., description="Paper URL")
    published_date: datetime | None = Field(
        default=None,
        description="Publication date",
    )
    categories: list[str] = Field(
        default_factory=list,
        description="ArXiv categories",
    )
    similarity_score: float | None = Field(
        default=None,
        description="Similarity score for vector search",
    )


class ResearchResult(BaseModel):
    """Research result containing papers and metadata."""

    papers: list[Paper] = Field(..., description="list of research papers")
    total_found: int = Field(..., description="Total number of papers found")
    search_query: str = Field(..., description="Original search query")
    search_time: float = Field(..., description="Search execution time in seconds")
    sources: list[str] = Field(
        default_factory=list,
        description="Sources used for research",
    )
    error: str | None = Field(default=None, description="Search execution time in seconds")


class ChatResponse(BaseModel):
    """Chat response model."""

    message: str = Field(..., description="Assistant response")
    conversation_id: str = Field(..., description="Conversation ID")
    research_results: ResearchResult | None = Field(
        default=None,
        description="Research results if applicable",
    )
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )


class StreamingChunk(BaseModel):
    """Streaming response chunk."""

    content: str = Field(..., description="Chunk content")
    chunk_type: str = Field(
        default="text",
        description="Type of chunk (text, paper, metadata)",
    )
    conversation_id: str = Field(..., description="Conversation ID")
    is_final: bool = Field(default=False, description="Is this the final chunk")


class WebSocketMessage(BaseModel):
    """WebSocket message model."""

    message: str = Field(..., description="User message")
    conversation_id: str | None = Field(
        default=None,
        description="Conversation ID",
    )
    action: str = Field(default="chat", description="Message action")


class WebSocketResponse(BaseModel):
    """WebSocket response model."""

    type: str = Field(..., description="Response type")
    content: str = Field(..., description="Response content")
    conversation_id: str = Field(..., description="Conversation ID")
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Additional metadata",
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = Field(..., description="Application version")
    uptime: float = Field(..., description="Uptime in seconds")


class PaperSearchRequest(BaseModel):
    """Paper search request model."""

    query: str = Field(..., description="Search query", min_length=1)
    limit: int = Field(
        default=10,
        description="Maximum number of results",
        ge=1,
        le=50,
    )
    similarity_threshold: float | None = Field(
        default=None,
        description="Minimum similarity threshold",
        ge=0.0,
        le=1.0,
    )


class PaperSearchResponse(BaseModel):
    """Paper search response model."""

    papers: list[Paper] = Field(..., description="Matching papers")
    total_found: int = Field(..., description="Total papers found")
    search_time: float = Field(..., description="Search time in seconds")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    detail: str | None = Field(default=None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.now)
