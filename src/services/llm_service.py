"""Shared LLM and embeddings service for the entire project."""

import traceback

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from loguru import logger

from ..di.fabric import create_embeddings_model_instance, create_llm_instance


class LLMService:
    """Shared LLM and embeddings service."""

    def __init__(self):
        self._chat_llm: BaseChatModel = create_llm_instance()
        self._embeddings_model: Embeddings = create_embeddings_model_instance()
        logger.info("LLM Service initialized")

    @property
    def chat_llm(self) -> BaseChatModel:
        return self._chat_llm

    @property
    def embeddings_model(self) -> Embeddings:
        return self._embeddings_model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        try:
            logger.debug(f"Embedding {len(texts)} documents")
            return self.embeddings_model.embed_documents(texts)
        except Exception:
            logger.error(f"Error embedding documents: {traceback.format_exc()}")
            raise

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        try:
            logger.debug(f"Embedding query: {text[:50]}...")
            return self.embeddings_model.embed_query(text)
        except Exception:
            logger.error(f"Error embedding query: {traceback.format_exc()}")
            raise

    def invoke_chat(self, prompt: str) -> str:
        """Invoke the chat LLM with a prompt."""
        try:
            logger.debug(f"Invoking chat LLM with prompt: {prompt[:100]}...")
            response = self.chat_llm.invoke(prompt)
            return response.content.strip()
        except Exception:
            logger.error(f"Error invoking chat LLM: {traceback.format_exc()}")
            raise

    async def ainvoke_chat(self, prompt: str) -> str:
        """Async invoke the chat LLM with a prompt."""
        try:
            logger.debug(f"Async invoking chat LLM with prompt: {prompt[:100]}...")
            response = await self.chat_llm.ainvoke(prompt)
            return response.content.strip()
        except Exception:
            logger.error(f"Error async invoking chat LLM: {traceback.format_exc()}")
            raise


# Global LLM service instance
llm_service = LLMService()
