from functools import lru_cache

from pipeline.factory import create_pipeline

from .service import ChatService


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    """Carga y entrena el pipeline una sola vez por proceso del API."""
    pipeline = create_pipeline()
    pipeline.run()
    return ChatService(pipeline)
