from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ..config.settings import settings, ModelType


def create_llm_instance() -> BaseChatModel:
    match settings.model.model_type:
        case ModelType.openai:
            return ChatOpenAI(
                model=settings.model.openai.model,
                api_key=settings.model.openai.api_key,
                temperature=0.3,
            )
        case _:
            raise NotImplemented(f"Model {settings.model.model_type} is not supported.")


def create_embeddings_model_instance() -> Embeddings:
    match settings.model.model_type:
        case ModelType.openai:
            return OpenAIEmbeddings(
                model=settings.model.openai.embedding_model,
                api_key=settings.model.openai.api_key,
            )
        case _:
            raise NotImplemented(f"Model {settings.model.model_type} is not supported.")
