import logging
from typing import Dict, Optional

import numpy as np
from scipy.sparse import spmatrix
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

from ..config.wine_clustering_config import WineClusteringConfig
from ..contracts.unsupervised_clusterer import UnsupervisedClusterer

logger = logging.getLogger("WinePipeline.ClusteringLayer")


class WineMiniBatchKMeansClusterer(UnsupervisedClusterer):
    """Agrupa perfiles de vinos mediante MiniBatchKMeans y calcula métricas internas."""

    def __init__(self, config: Optional[WineClusteringConfig] = None) -> None:
        self._config = config or WineClusteringConfig()
        self._model = MiniBatchKMeans(
            n_clusters=self._config.n_clusters,
            batch_size=self._config.batch_size,
            random_state=self._config.random_state,
            n_init="auto",
        )
        self._is_fitted: bool = False
        self._metrics: Dict[str, Optional[float]] = {}

    @property
    def metrics(self) -> Dict[str, Optional[float]]:
        """Retorna las métricas de evaluación no supervisada calculadas."""
        return dict(self._metrics)

    def __repr__(self) -> str:
        return (
            f"WineMiniBatchKMeansClusterer(n_clusters={self._config.n_clusters}, "
            f"is_fitted={self._is_fitted})"
        )

    def _validate_matrix(self, features: spmatrix) -> None:
        if features is None or features.shape[0] == 0:
            raise ValueError("La matriz de características está vacía.")
        if features.shape[0] < self._config.n_clusters:
            raise ValueError(
                f"Hay menos muestras ({features.shape[0]}) que clústeres ({self._config.n_clusters})."
            )

    def fit_predict(self, features: spmatrix) -> np.ndarray:
        self._validate_matrix(features)
        try:
            labels = self._model.fit_predict(features)
        except Exception as exc:
            raise RuntimeError(f"Error durante el entrenamiento de clustering: {exc}") from exc

        self._is_fitted = True
        self._compute_metrics(features, labels)
        logger.info("Clustering completado. Métricas: %s", self._metrics)
        return labels

    def predict(self, features: spmatrix) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("El modelo de clustering debe entrenarse antes de llamar a predict().")
        if features is None or features.shape[0] == 0:
            raise ValueError("La matriz de entrada para predict() está vacía.")
        return self._model.predict(features)

    def _compute_metrics(self, features: spmatrix, labels: np.ndarray) -> None:
        inertia = float(self._model.inertia_)
        silhouette: Optional[float] = None

        unique_labels = np.unique(labels)
        if len(unique_labels) > 1 and features.shape[0] > len(unique_labels):
            sample_size = min(self._config.silhouette_sample_size, features.shape[0])
            silhouette = float(
                silhouette_score(
                    features,
                    labels,
                    sample_size=sample_size,
                    random_state=self._config.random_state,
                )
            )

        self._metrics = {
            "inertia": inertia,
            "silhouette_score": silhouette,
        }
