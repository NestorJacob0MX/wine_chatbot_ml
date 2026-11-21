import asyncio
from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile, status

from .runtime import get_chat_service
from .schemas import (
    ChatMessageRequest,
    ChatMessageResponse,
    ChatSessionResponse,
    HealthResponse,
    ImageMessageResponse,
)

MAX_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
logger = logging.getLogger("WinePipeline.API")


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
    service = get_chat_service()
    return HealthResponse(
        status="ok",
        pipeline_ready=True,
        image_classifier_ready=service.image_classifier_ready,
    )


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


@app.post(
    "/api/sessions/{session_id}/images",
    response_model=ImageMessageResponse,
)
async def send_image(
    session_id: str,
    image: UploadFile = File(...),
    caption: str = Form(default=""),
) -> ImageMessageResponse:
    try:
        if image.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=415,
                detail="Formato no compatible. Envía JPG, PNG o WebP.",
            )
        image_bytes = await image.read(MAX_IMAGE_BYTES + 1)
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=413,
                detail="La imagen excede el límite de 10 MB.",
            )
        if caption.strip():
            logger.info("Imagen recibida con comentario: %s", caption.strip()[:300])
    finally:
        await image.close()

    service = get_chat_service()
    try:
        result = await asyncio.to_thread(service.analyze_image, session_id, image_bytes)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=exc.args[0]) from exc
    except RuntimeError as exc:
        status_code = 503 if not service.image_classifier_ready else 409
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    message, food_label, confidence, closed, currency = result
    return ImageMessageResponse(
        message=message,
        food_label=food_label,
        confidence=confidence,
        closed=closed,
        currency=currency,
    )


@app.delete(
    "/api/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
def close_session(session_id: str) -> Response:
    if not get_chat_service().close_session(session_id):
        raise HTTPException(status_code=404, detail="No existe la sesión solicitada.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
