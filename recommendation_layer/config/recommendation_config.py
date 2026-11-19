from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RecommendationConfig:
    """Configuración para recomendar vinos a partir de una consulta o cena."""

    top_k: int = 3
    cluster_col: str = "cluster_id"
    min_points: Optional[float] = None
    max_price: Optional[float] = None
    similarity_weight: float = 0.75
    quality_weight: float = 0.25
