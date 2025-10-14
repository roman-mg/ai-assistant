from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ..config.settings import settings


def create_llm_instance() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.model.openai.model,
        api_key=settings.model.openai.api_key,
        temperature=0.3,
    )


def create_embeddings_model_instance() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.model.openai.embedding_model,
        api_key=settings.model.openai.api_key,
    )
