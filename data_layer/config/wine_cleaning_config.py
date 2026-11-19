from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class WineCleaningConfig:
    """Configuración explícita para la carga y limpieza del dataset de vinos."""

    required_columns: Tuple[str, ...] = ("description", "points", "price", "variety")
    optional_metadata_columns: Tuple[str, ...] = ("title", "winery")
    min_points: float = 70.0
    max_points: float = 100.0
    min_price: float = 1.0
    max_price: float = 5000.0
    impute_missing_price: bool = True
    drop_duplicates_subset: Tuple[str, ...] = ("description", "title")
    text_columns: Tuple[str, ...] = ("description", "variety", "title", "winery")
    default_extra_cols: List[str] = field(default_factory=list)

    @property
    def active_columns(self) -> List[str]:
        cols = list(self.required_columns) + list(self.optional_metadata_columns)
        for col in self.default_extra_cols:
            if col not in cols:
                cols.append(col)
        return cols
