from abc import ABC, abstractmethod

import pandas as pd


class DataCleaner(ABC):
    """Contrato para limpiadores de datos."""

    @abstractmethod
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica limpieza y validación sobre un DataFrame."""
