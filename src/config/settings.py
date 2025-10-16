"""Configuration management using Pydantic settings."""

from enum import StrEnum

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelType(StrEnum):
    openai = "openai"
    ollama = "ollama"


class OpenAISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OPENAI_", env_file=".env", extra="ignore", case_sensitive=False)
    api_key: str = Field(
        default="test",
        description="OpenAI API key for LLM and embeddings",
    )
    model: str = Field(
        default="gpt-4o",
        description="OpenAI model to use for chat completions",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI model to use for embeddings",
    )


class OllamaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OPENAI_", env_file=".env", extra="ignore", case_sensitive=False)
    model: str = Field(
        default="phi",
        description="Ollama model to use for chat completions",
    )
    embedding_model: str = Field(
        default="all-minilm",
        description="Ollama model to use for embeddings",
    )


class ModelSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MODEL_", env_file=".env", extra="ignore", case_sensitive=False)
    type: ModelType = Field(
        default="ollama",
        description="Specify models set for LLM and embeddings",
    )
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)


class ApplicationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env", extra="ignore", case_sensitive=False)
    name: str = Field(
        default="AI Research Assistant",
        description="Application name",
    )
    version: str = Field(
        default="0.1.0",
        description="Application version",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )


class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SERVER_", env_file=".env", extra="ignore", case_sensitive=False)
    host: str = Field(
        default="0.0.0.0",
        description="Server host",
    )
    port: int = Field(
        default=8000,
        description="Server port",
    )


class ResearcherSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RESEARCHER_", env_file=".env", extra="ignore", case_sensitive=False)
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
    graph_diagram_path: str = Field(
        default="./data/graph.png",
        description="Path to store LangGraph diagram",
    )


class VectorStoreSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VECTOR_STORE_", env_file=".env", extra="ignore", case_sensitive=False)
    faiss_index_path: str = Field(
        default="./data/faiss_index",
        description="Path to store FAISS index",
    )
    vector_dimension: int = Field(
        default=384,
        description="Dimension of embedding vectors",
    )
    similarity_threshold: float = Field(
        default=0.7,
        description="Minimum similarity threshold for vector search",
    )


class WebSearchSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WEB_SEARCH_", env_file=".env", extra="ignore", case_sensitive=False)
    enabled: bool = Field(
        default=True,
        description="Enable web search functionality",
    )
    max_results: int = Field(
        default=5,
        description="Maximum web search results to process",
    )
    search_engine: str = Field(
        default="duckduckgo",
        description="Search engine to use",
    )


class ConversationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CONVERSATION_", env_file=".env", extra="ignore", case_sensitive=False)
    max_history: int = Field(
        default=10,
        description="Maximum number of messages to keep in conversation history",
    )
    timeout: int = Field(
        default=3600,
        description="Conversation timeout in seconds",
    )


class Settings(BaseSettings):
    """Application settings with default values."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)
    app: ApplicationSettings = Field(default_factory=ApplicationSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    researcher: ResearcherSettings = Field(default_factory=ResearcherSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    web_search: WebSearchSettings = Field(default_factory=WebSearchSettings)
    conversation: ConversationSettings = Field(default_factory=ConversationSettings)


load_dotenv()
settings = Settings()
