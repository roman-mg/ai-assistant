"""Shared LLM and embeddings service for the entire project."""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loguru import logger

from ..config.settings import settings


class LLMService:
    """Shared LLM and embeddings service."""

    def __init__(self):
        self._chat_llm: ChatOpenAI = ChatOpenAI(
            model=settings.openai.model,
            api_key=settings.openai.api_key,
            temperature=0.3,
        )
        self._embeddings_model: OpenAIEmbeddings = OpenAIEmbeddings(
            model=settings.openai.embedding_model,
            api_key=settings.openai.api_key,
        )
        logger.info("LLM Service initialized")

    @property
    def chat_llm(self) -> ChatOpenAI:
        return self._chat_llm

    @property
    def embeddings_model(self) -> OpenAIEmbeddings:
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


# Global LLM service instance
llm_service = LLMService()
