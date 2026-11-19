from dataclasses import dataclass


@dataclass(frozen=True)
class WineRecommendation:
    """Estructura de salida interpretable para una recomendación de vino."""

    title: str
    variety: str
    winery: str
    price: float
    points: float
    cluster_id: int
    similarity_score: float
    final_score: float
    description: str
