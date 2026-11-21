from pydantic import BaseModel, Field


class ChatMessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class ChatSessionResponse(BaseModel):
    session_id: str


class ChatMessageResponse(BaseModel):
    message: str
    closed: bool
    currency: str


class ImageMessageResponse(BaseModel):
    message: str
    food_label: str
    confidence: float
    closed: bool
    currency: str


class HealthResponse(BaseModel):
    status: str
    pipeline_ready: bool
    image_classifier_ready: bool
