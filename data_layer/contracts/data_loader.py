from abc import ABC, abstractmethod

import pandas as pd


class DataLoader(ABC):
    """Contrato para cargadores de datos."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Carga y retorna un DataFrame."""
