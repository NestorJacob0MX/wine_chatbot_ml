import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from ..contracts.data_loader import DataLoader

logger = logging.getLogger("WinePipeline.DataLayer")


class CsvWineDataLoader(DataLoader):
    """Carga el dataset CSV de reseñas de vinos y optimiza tipos básicos."""

    def __init__(self, filepath: str | Path, columns: Optional[List[str]] = None) -> None:
        self._filepath = Path(filepath)
        self._columns = columns
        self._data: Optional[pd.DataFrame] = None

    def __len__(self) -> int:
        return 0 if self._data is None else len(self._data)

    def __repr__(self) -> str:
        loaded_rows = len(self)
        return f"CsvWineDataLoader(filepath={str(self._filepath)!r}, rows_loaded={loaded_rows})"

    def load(self) -> pd.DataFrame:
        self._validate_file_exists()
        try:
            if self._columns:
                available_cols = self._get_available_columns()
                df = pd.read_csv(self._filepath, usecols=available_cols, low_memory=False)
            else:
                df = pd.read_csv(self._filepath, low_memory=False)
        except Exception as exc:
            raise RuntimeError(f"Fallo al leer el archivo CSV '{self._filepath}': {exc}") from exc

        df = self._optimize_numeric_dtypes(df)
        self._data = df
        logger.info("Dataset cargado con forma %s desde '%s'", df.shape, self._filepath)
        return df.copy()

    def _validate_file_exists(self) -> None:
        if not self._filepath.exists() or not self._filepath.is_file():
            raise FileNotFoundError(f"No se encontró el archivo de datos en: {self._filepath}")

    def _get_available_columns(self) -> List[str]:
        header = pd.read_csv(self._filepath, nrows=0)
        existing = [col for col in (self._columns or []) if col in header.columns]
        missing = [col for col in (self._columns or []) if col not in header.columns]
        if missing:
            logger.warning("Columnas solicitadas no encontradas en el CSV (se omiten): %s", missing)
        if not existing:
            raise ValueError("Ninguna de las columnas solicitadas existe en el archivo CSV.")
        return existing

    @staticmethod
    def _optimize_numeric_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        optimized = df.copy()
        if "points" in optimized.columns:
            optimized["points"] = pd.to_numeric(optimized["points"], errors="coerce").astype("float32")
        if "price" in optimized.columns:
            optimized["price"] = pd.to_numeric(optimized["price"], errors="coerce").astype("float32")
        return optimized
