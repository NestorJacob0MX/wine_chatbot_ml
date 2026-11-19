import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.metrics.pairwise import cosine_similarity

from nlp_layer import TextVectorizer
from ..config.recommendation_config import RecommendationConfig
from ..contracts.wine_recommender import WineRecommender
from ..models.wine_recommendation import WineRecommendation

logger = logging.getLogger("WinePipeline.RecommendationLayer")


class ContentClusterWineRecommender(WineRecommender):
    """Recomienda vinos combinando similitud TF-IDF, clúster afín y puntaje."""

    def __init__(
        self,
        catalog_df: pd.DataFrame,
        item_features: spmatrix,
        vectorizer: TextVectorizer,
        cluster_terms: Optional[Dict[int, List[str]]] = None,
        config: Optional[RecommendationConfig] = None,
    ) -> None:
        self._validate_inputs(catalog_df, item_features, vectorizer)
        self._catalog_df = catalog_df.reset_index(drop=True).copy()
        self._item_features = item_features
        self._vectorizer = vectorizer
        self._cluster_terms = cluster_terms or {}
        self._config = config or RecommendationConfig()

    @staticmethod
    def _validate_inputs(
        catalog_df: pd.DataFrame,
        item_features: spmatrix,
        vectorizer: TextVectorizer,
    ) -> None:
        if not isinstance(catalog_df, pd.DataFrame) or catalog_df.empty:
            raise ValueError("El catálogo de vinos no puede estar vacío.")
        if item_features is None or item_features.shape[0] != len(catalog_df):
            raise ValueError(
                "La matriz de características debe tener el mismo número de filas que el catálogo."
            )
        if vectorizer is None:
            raise ValueError("Se requiere un vectorizador entrenado.")

    def recommend(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        max_price: Optional[float] = None,
    ) -> List[WineRecommendation]:
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError("La descripción de la cena o preferencia no puede estar vacía.")

        k = top_k if top_k is not None else self._config.top_k
        if k <= 0:
            raise ValueError("top_k debe ser mayor que cero.")

        query_vec = self._vectorizer.transform_query(query_text)
        similarities = cosine_similarity(query_vec, self._item_features).ravel()

        candidate_mask = np.ones(len(self._catalog_df), dtype=bool)
        effective_max_price = (
            max_price if max_price is not None else self._config.max_price
        )

        if effective_max_price is not None and "price" in self._catalog_df.columns:
            candidate_mask &= self._catalog_df["price"].to_numpy(dtype=float) <= float(
                effective_max_price
            )

        if self._config.min_points is not None and "points" in self._catalog_df.columns:
            candidate_mask &= self._catalog_df["points"].to_numpy(dtype=float) >= float(
                self._config.min_points
            )

        candidate_indices = np.flatnonzero(candidate_mask)
        if candidate_indices.size == 0:
            logger.warning(
                "No hubo vinos que cumplieran los filtros; se evalúa todo el catálogo."
            )
            candidate_indices = np.arange(len(self._catalog_df))

        scores = self._compute_final_scores(candidate_indices, similarities)
        ranked_order = np.argsort(scores)[::-1][:k]
        selected_indices = candidate_indices[ranked_order]

        recommendations: List[WineRecommendation] = []
        for rank, idx in enumerate(selected_indices):
            row = self._catalog_df.iloc[int(idx)]
            recommendations.append(
                WineRecommendation(
                    title=str(row.get("title", "Vino sin título")),
                    variety=str(row.get("variety", "Variedad no especificada")),
                    winery=str(row.get("winery", "Bodega no especificada")),
                    price=float(row.get("price", 0.0)),
                    points=float(row.get("points", 0.0)),
                    cluster_id=int(row.get(self._config.cluster_col, -1)),
                    similarity_score=float(similarities[int(idx)]),
                    final_score=float(scores[ranked_order[rank]]),
                    description=str(row.get("description", "")),
                )
            )

        logger.info(
            "Generadas %d recomendaciones para consulta: '%s'",
            len(recommendations),
            query_text,
        )
        return recommendations

    def _compute_final_scores(
        self,
        candidate_indices: np.ndarray,
        similarities: np.ndarray,
    ) -> np.ndarray:
        sim_subset = similarities[candidate_indices]

        if "points" in self._catalog_df.columns:
            points_subset = self._catalog_df["points"].iloc[candidate_indices].to_numpy(
                dtype=float
            )
            p_min, p_max = float(np.min(points_subset)), float(np.max(points_subset))
            norm_points = (
                (points_subset - p_min) / (p_max - p_min)
                if p_max > p_min
                else np.ones_like(points_subset)
            )
        else:
            norm_points = np.zeros_like(sim_subset)

        return (
            self._config.similarity_weight * sim_subset
            + self._config.quality_weight * norm_points
        )
