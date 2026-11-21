from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class FoodPrediction:
    label: str
    confidence: float


class FoodClassifier(Protocol):
    """Contrato para clasificadores de imágenes de alimentos."""

    def predict(self, image_bytes: bytes) -> FoodPrediction:
        """Predice la clase de alimento a partir de bytes de imagen."""
