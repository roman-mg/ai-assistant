from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama, OllamaEmbeddings

from ..config.settings import ModelType, settings


def create_llm_instance() -> BaseChatModel:
    match settings.model.type:
        case ModelType.openai:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=settings.model.openai.model,
                api_key=settings.model.openai.api_key,
                temperature=0.3,
            )
        case ModelType.ollama:
            return ChatOllama(
                model=settings.model.ollama.model,
                temperature=0.3,
            )
        case _:
            raise NotImplementedError(f"Model {settings.model.type} is not supported.")


def create_embeddings_model_instance() -> Embeddings:
    match settings.model.type:
        case ModelType.openai:
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(
                model=settings.model.openai.embedding_model,
                api_key=settings.model.openai.api_key,
            )
        case ModelType.ollama:
            return OllamaEmbeddings(model=settings.model.ollama.embedding_model)
        case _:
            raise NotImplementedError(f"Model {settings.model.type} is not supported.")
