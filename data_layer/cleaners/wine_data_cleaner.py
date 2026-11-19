import logging
from typing import Optional

import pandas as pd

from ..config.wine_cleaning_config import WineCleaningConfig
from ..contracts.data_cleaner import DataCleaner

logger = logging.getLogger("WinePipeline.DataLayer")


class WineDataCleaner(DataCleaner):
    """Limpia, valida y estandariza los datos de reseñas de vinos."""

    def __init__(self, config: Optional[WineCleaningConfig] = None) -> None:
        self._config = config or WineCleaningConfig()
        self._last_report: dict = {}

    @property
    def last_report(self) -> dict:
        """Retorna un resumen de auditoría de la última ejecución de limpieza."""
        return dict(self._last_report)

    def __repr__(self) -> str:
        return f"WineDataCleaner(config={self._config!r})"

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate_input_dataframe(df)
        initial_rows = len(df)

        cleaned = df.copy()
        cleaned = self._select_relevant_columns(cleaned)
        cleaned = self._clean_text_fields(cleaned)
        cleaned = self._drop_invalid_core_records(cleaned)
        cleaned = self._filter_valid_ranges(cleaned)
        cleaned = self._handle_missing_prices(cleaned)
        cleaned = self._remove_duplicates(cleaned)
        cleaned = cleaned.reset_index(drop=True)

        self._last_report = {
            "initial_rows": initial_rows,
            "final_rows": len(cleaned),
            "removed_rows": initial_rows - len(cleaned),
            "columns": list(cleaned.columns),
            "null_counts": cleaned.isna().sum().to_dict(),
        }
        logger.info(
            "Limpieza completada: %d -> %d filas (eliminadas: %d)",
            initial_rows,
            len(cleaned),
            initial_rows - len(cleaned),
        )
        return cleaned

    def _validate_input_dataframe(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Se esperaba un pd.DataFrame, se recibió {type(df).__name__}")
        if df.empty:
            raise ValueError("El DataFrame de entrada está vacío.")
        missing_required = [
            col for col in self._config.required_columns if col not in df.columns
        ]
        if missing_required:
            raise KeyError(f"Faltan columnas obligatorias en el dataset: {missing_required}")

    def _select_relevant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        selected = [col for col in self._config.active_columns if col in df.columns]
        return df[selected].copy()

    def _clean_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self._config.text_columns:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype("string")
                    .str.strip()
                    .str.replace(r"\s+", " ", regex=True)
                )
                df.loc[df[col] == "", col] = pd.NA
        return df

    def _drop_invalid_core_records(self, df: pd.DataFrame) -> pd.DataFrame:
        core_non_null = [
            col for col in ("description", "variety", "points") if col in df.columns
        ]
        return df.dropna(subset=core_non_null)

    def _filter_valid_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        if "points" in df.columns:
            valid_points = df["points"].between(
                self._config.min_points, self._config.max_points, inclusive="both"
            )
            df = df[valid_points]

        if "price" in df.columns:
            valid_price = df["price"].isna() | df["price"].between(
                self._config.min_price, self._config.max_price, inclusive="both"
            )
            df = df[valid_price]

        return df

    def _handle_missing_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        if "price" not in df.columns or not self._config.impute_missing_price:
            return df

        if df["price"].isna().any():
            if "variety" in df.columns:
                variety_medians = df.groupby("variety")["price"].transform("median")
                global_median = df["price"].median()
                df["price"] = df["price"].fillna(variety_medians).fillna(global_median)
            else:
                df["price"] = df["price"].fillna(df["price"].median())

        df["price"] = df["price"].astype("float32")
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        subset = [
            col for col in self._config.drop_duplicates_subset if col in df.columns
        ]
        if not subset:
            return df
        return df.drop_duplicates(subset=subset, keep="first")
