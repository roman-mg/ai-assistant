"""Shared LLM and embeddings service for the entire project."""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loguru import logger

from src.config.settings import settings


class LLMService:
    """Shared LLM and embeddings service."""

    def __init__(self):
        """Initialize the LLM service with shared instances."""
        self._chat_llm: ChatOpenAI | None = None
        self._embeddings_model: OpenAIEmbeddings | None = None
        logger.info("LLM Service initialized")

    @property
    def chat_llm(self) -> ChatOpenAI:
        """Get the shared chat LLM instance."""
        if self._chat_llm is None:
            self._chat_llm = ChatOpenAI(
                model=settings.openai_model,
                api_key=settings.openai_api_key,
                temperature=0.3,
            )
            logger.info(f"Created shared chat LLM: {settings.openai_model}")
        return self._chat_llm

    @property
    def embeddings_model(self) -> OpenAIEmbeddings:
        """Get the shared embeddings model instance."""
        if self._embeddings_model is None:
            self._embeddings_model = OpenAIEmbeddings(
                model=settings.openai_embedding_model,
                api_key=settings.openai_api_key,
            )
            logger.info(f"Created shared embeddings model: {settings.openai_embedding_model}")
        return self._embeddings_model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        try:
            logger.debug(f"Embedding {len(texts)} documents")
            return self.embeddings_model.embed_documents(texts)
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            raise

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        try:
            logger.debug(f"Embedding query: {text[:50]}...")
            return self.embeddings_model.embed_query(text)
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise

    def invoke_chat(self, prompt: str) -> str:
        """Invoke the chat LLM with a prompt."""
        try:
            logger.debug(f"Invoking chat LLM with prompt: {prompt[:100]}...")
            response = self.chat_llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error invoking chat LLM: {e}")
            raise

    async def ainvoke_chat(self, prompt: str) -> str:
        """Async invoke the chat LLM with a prompt."""
        try:
            logger.debug(f"Async invoking chat LLM with prompt: {prompt[:100]}...")
            response = await self.chat_llm.ainvoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error async invoking chat LLM: {e}")
            raise

    def reset(self) -> None:
        """Reset the LLM instances (useful for testing)."""
        self._chat_llm = None
        self._embeddings_model = None
        logger.info("LLM Service reset")


# Global LLM service instance
llm_service = LLMService()
