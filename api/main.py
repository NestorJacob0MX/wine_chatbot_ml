import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Response, status

from .runtime import get_chat_service
from .schemas import (
    ChatMessageRequest,
    ChatMessageResponse,
    ChatSessionResponse,
    HealthResponse,
)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    await asyncio.to_thread(get_chat_service)
    yield


app = FastAPI(
    title="Wine Chatbot API",
    description="API local para el chatbot de recomendación de vinos.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    get_chat_service()
    return HealthResponse(status="ok", pipeline_ready=True)


@app.post(
    "/api/sessions",
    response_model=ChatSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_session() -> ChatSessionResponse:
    return ChatSessionResponse(session_id=get_chat_service().create_session())


@app.post(
    "/api/sessions/{session_id}/messages",
    response_model=ChatMessageResponse,
)
def send_message(
    session_id: str,
    payload: ChatMessageRequest,
) -> ChatMessageResponse:
    try:
        message, closed, currency = get_chat_service().send_message(
            session_id,
            payload.message,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=exc.args[0]) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return ChatMessageResponse(message=message, closed=closed, currency=currency)


@app.delete(
    "/api/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
def close_session(session_id: str) -> Response:
    if not get_chat_service().close_session(session_id):
        raise HTTPException(status_code=404, detail="No existe la sesión solicitada.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
