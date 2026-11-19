from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix


class TextVectorizer(ABC):
    """Contrato para componentes que transforman texto en representación vectorial."""

    @abstractmethod
    def fit_transform(self, df: pd.DataFrame) -> spmatrix:
        """Ajusta el vectorizador y transforma reseñas en una matriz de características."""

    @abstractmethod
    def transform_query(self, query_text: str) -> spmatrix:
        """Transforma una consulta al mismo espacio vectorial."""

    @abstractmethod
    def get_top_terms_per_cluster(
        self,
        labels: np.ndarray,
        features: spmatrix,
        top_n: int = 8,
    ) -> Dict[int, List[str]]:
        """Obtiene los términos con mayor peso promedio en cada clúster."""
