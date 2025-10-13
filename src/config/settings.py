"""Configuration management using Pydantic settings."""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with default values."""

    # OpenAI Configuration
    openai_api_key: str = Field(
        default="sk-your-openai-api-key-here",
        description="OpenAI API key for LLM and embeddings",
    )
    openai_model: str = Field(
        default="gpt-4o",
        description="OpenAI model to use for chat completions",
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI model to use for embeddings",
    )

    # Application Configuration
    app_name: str = Field(
        default="AI Research Assistant",
        description="Application name",
    )
    app_version: str = Field(
        default="0.1.0",
        description="Application version",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )

    # Server Configuration
    host: str = Field(
        default="0.0.0.0",
        description="Server host",
    )
    port: int = Field(
        default=8000,
        description="Server port",
    )

    # Research Configuration
    max_papers_per_query: int = Field(
        default=10,
        description="Maximum number of papers to return per query",
    )
    max_paper_summary_length: int = Field(
        default=500,
        description="Maximum length of paper summaries",
    )
    arxiv_max_results: int = Field(
        default=50,
        description="Maximum results to fetch from ArXiv",
    )

    # Vector Store Configuration
    faiss_index_path: str = Field(
        default="./data/faiss_index",
        description="Path to store FAISS index",
    )
    vector_dimension: int = Field(
        default=1536,
        description="Dimension of embedding vectors",
    )
    similarity_threshold: float = Field(
        default=0.7,
        description="Minimum similarity threshold for vector search",
    )

    # Web Search Configuration
    web_search_enabled: bool = Field(
        default=True,
        description="Enable web search functionality",
    )
    max_web_results: int = Field(
        default=5,
        description="Maximum web search results to process",
    )

    # Conversation Configuration
    max_conversation_history: int = Field(
        default=10,
        description="Maximum number of messages to keep in conversation history",
    )
    conversation_timeout: int = Field(
        default=3600,
        description="Conversation timeout in seconds",
    )

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
