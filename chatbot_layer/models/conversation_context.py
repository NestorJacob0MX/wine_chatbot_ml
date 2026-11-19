from dataclasses import dataclass
from typing import Optional


@dataclass
class ConversationContext:
    """Estado ligero de la conversación para enriquecer recomendaciones."""

    language: str = "es"
    last_dinner_query: Optional[str] = None
    max_price: Optional[float] = None
    intent: Optional[str] = None
    awaiting: Optional[str] = None
