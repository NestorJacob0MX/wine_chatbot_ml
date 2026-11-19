from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class WineClusteringConfig:
    """Configuración para vectorización TF-IDF y clustering no supervisado."""

    text_column: str = "description"
    variety_column: str = "variety"
    include_variety_in_text: bool = True
    max_features: int = 3000
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 5
    max_df: float = 0.85
    stop_words: Optional[str] = "english"
    n_clusters: int = 6
    batch_size: int = 1024
    random_state: int = 42
    cluster_col_name: str = "cluster_id"
    silhouette_sample_size: int = 5000
