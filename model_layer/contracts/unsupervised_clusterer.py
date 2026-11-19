from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np
from scipy.sparse import spmatrix


class UnsupervisedClusterer(ABC):
    """Contrato para modelos de agrupamiento no supervisado."""

    @property
    @abstractmethod
    def metrics(self) -> Dict[str, Optional[float]]:
        """Métricas calculadas durante el ajuste del modelo."""

    @abstractmethod
    def fit_predict(self, features: spmatrix) -> np.ndarray:
        """Entrena el modelo de clustering y retorna las etiquetas de clúster."""

    @abstractmethod
    def predict(self, features: spmatrix) -> np.ndarray:
        """Asigna clústeres a nuevas muestras vectorizadas."""
