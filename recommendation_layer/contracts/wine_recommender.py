from abc import ABC, abstractmethod
from typing import List, Optional

from ..models.wine_recommendation import WineRecommendation


class WineRecommender(ABC):
    """Contrato para motores de recomendación de vinos."""

    @abstractmethod
    def recommend(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        max_price: Optional[float] = None,
    ) -> List[WineRecommendation]:
        """Retorna una lista ordenada de vinos recomendados para la consulta."""
