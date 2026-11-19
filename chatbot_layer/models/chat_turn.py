from dataclasses import dataclass
from typing import Tuple

from recommendation_layer.models.wine_recommendation import WineRecommendation


@dataclass(frozen=True)
class ChatTurn:
    """Representa un turno del historial de conversación."""

    role: str
    message: str
    recommendations: Tuple[WineRecommendation, ...] = ()
