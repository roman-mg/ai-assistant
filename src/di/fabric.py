from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

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
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=settings.model.ollama.model,
                temperature=0.3,
            )
        # TODO: add vLLM support
        # case ModelType.vllm:
        #     from langchain_community.llms import VLLM
        #
        #     llm = VLLM(
        #         model="mistralai/Mistral-7B-Instruct-v0.3",
        #         base_url="http://localhost:8000/v1",
        #     )
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
            from langchain_ollama import OllamaEmbeddings

            return OllamaEmbeddings(model=settings.model.ollama.embedding_model)
        case _:
            raise NotImplementedError(f"Model {settings.model.type} is not supported.")
