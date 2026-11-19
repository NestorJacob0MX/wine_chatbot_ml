import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

from model_layer.config.wine_clustering_config import WineClusteringConfig
from ..contracts.text_vectorizer import TextVectorizer

logger = logging.getLogger("WinePipeline.ClusteringLayer")


class TfidfWineVectorizer(TextVectorizer):
    """Construye representaciones TF-IDF a partir de las reseñas de vinos."""

    def __init__(self, config: Optional[WineClusteringConfig] = None) -> None:
        self._config = config or WineClusteringConfig()
        self._vectorizer = TfidfVectorizer(
            max_features=self._config.max_features,
            ngram_range=self._config.ngram_range,
            min_df=self._config.min_df,
            max_df=self._config.max_df,
            stop_words=self._config.stop_words,
        )
        self._is_fitted: bool = False

    def __repr__(self) -> str:
        return (
            f"TfidfWineVectorizer(max_features={self._config.max_features}, "
            f"n_clusters={self._config.n_clusters}, is_fitted={self._is_fitted})"
        )

    def _build_corpus(self, df: pd.DataFrame) -> List[str]:
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("El DataFrame de entrada no puede estar vacío.")
        if self._config.text_column not in df.columns:
            raise KeyError(f"Falta la columna de texto '{self._config.text_column}'.")

        descriptions = df[self._config.text_column].fillna("").astype(str)

        if self._config.include_variety_in_text and self._config.variety_column in df.columns:
            varieties = df[self._config.variety_column].fillna("").astype(str)
            corpus = (varieties + " " + descriptions).str.strip()
        else:
            corpus = descriptions.str.strip()

        non_empty = corpus[corpus != ""]
        if non_empty.empty:
            raise ValueError("No hay textos válidos para vectorizar tras la preparación.")

        return corpus.tolist()

    def fit_transform(self, df: pd.DataFrame) -> spmatrix:
        corpus = self._build_corpus(df)
        try:
            matrix = self._vectorizer.fit_transform(corpus)
        except Exception as exc:
            raise RuntimeError(f"Error al ajustar y transformar TF-IDF: {exc}") from exc

        self._is_fitted = True
        logger.info("Matriz TF-IDF generada con forma %s", matrix.shape)
        return matrix

    def transform_query(self, query_text: str) -> spmatrix:
        if not self._is_fitted:
            raise RuntimeError("El vectorizador debe entrenarse antes de transformar consultas.")
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError("La consulta del usuario debe ser un texto no vacío.")
        return self._vectorizer.transform([query_text.strip()])

    def get_top_terms_per_cluster(
        self,
        labels: np.ndarray,
        features: spmatrix,
        top_n: int = 8,
    ) -> Dict[int, List[str]]:
        """Obtiene los términos con mayor peso promedio en cada clúster."""
        if not self._is_fitted:
            raise RuntimeError("El vectorizador no ha sido ajustado.")
        if top_n <= 0:
            raise ValueError("top_n debe ser mayor que cero.")

        feature_names = np.asarray(self._vectorizer.get_feature_names_out())
        summaries: Dict[int, List[str]] = {}

        for cluster_id in sorted(np.unique(labels)):
            mask = labels == cluster_id
            if not np.any(mask):
                summaries[int(cluster_id)] = []
                continue
            mean_weights = np.asarray(features[mask].mean(axis=0)).ravel()
            top_indices = mean_weights.argsort()[::-1][:top_n]
            summaries[int(cluster_id)] = [str(feature_names[idx]) for idx in top_indices]

        return summaries
