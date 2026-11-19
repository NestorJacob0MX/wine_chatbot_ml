from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
from scipy.sparse import spmatrix

from data_layer import (
    CsvWineDataLoader,
    DataCleaner,
    DataLoader,
    StepTimer,
    WineCleaningConfig,
    WineDataCleaner,
)
from model_layer import (
    UnsupervisedClusterer,
    WineClusteringConfig,
    WineMiniBatchKMeansClusterer,
)
from nlp_layer import TextVectorizer, TfidfWineVectorizer


@dataclass(frozen=True)
class WinePipelineResult:
    """Resultados del flujo de carga, preparación, vectorización y agrupamiento."""

    dataframe: pd.DataFrame
    feature_matrix: spmatrix
    cluster_terms: Dict[int, List[str]]
    cluster_metrics: Dict[str, float | None]


class WineChatbotPipeline:
    """Orquesta las capas de datos, NLP y modelado no supervisado."""

    def __init__(
        self,
        data_loader: DataLoader,
        data_cleaner: DataCleaner,
        text_vectorizer: TextVectorizer,
        clusterer: UnsupervisedClusterer,
        cluster_col_name: str = "cluster_id",
    ) -> None:
        self._data_loader = data_loader
        self._data_cleaner = data_cleaner
        self._text_vectorizer = text_vectorizer
        self._clusterer = clusterer
        self._cluster_col_name = cluster_col_name

    def run(self) -> WinePipelineResult:
        with StepTimer("Capa de datos - Carga y limpieza"):
            raw_df = self._data_loader.load()
            clean_df = self._data_cleaner.clean(raw_df)

        with StepTimer("Capa NLP - Vectorización TF-IDF"):
            features = self._text_vectorizer.fit_transform(clean_df)

        with StepTimer("Capa de modelo - Clustering no supervisado"):
            labels = self._clusterer.fit_predict(features)

            result_df = clean_df.copy()
            result_df[self._cluster_col_name] = labels.astype(int)

        cluster_terms = self._text_vectorizer.get_top_terms_per_cluster(
            labels,
            features,
            top_n=8,
        )

        return WinePipelineResult(
            dataframe=result_df,
            feature_matrix=features,
            cluster_terms=cluster_terms,
            cluster_metrics=self._clusterer.metrics,
        )


def main() -> None:
    dataset_path = Path(__file__).resolve().parents[1] / "dataset" / "wines.csv"
    cleaning_config = WineCleaningConfig()
    clustering_config = WineClusteringConfig()

    pipeline = WineChatbotPipeline(
        data_loader=CsvWineDataLoader(
            filepath=dataset_path,
            columns=cleaning_config.active_columns,
        ),
        data_cleaner=WineDataCleaner(config=cleaning_config),
        text_vectorizer=TfidfWineVectorizer(config=clustering_config),
        clusterer=WineMiniBatchKMeansClusterer(config=clustering_config),
        cluster_col_name=clustering_config.cluster_col_name,
    )
    result = pipeline.run()

    print("\n--- Métricas de clustering ---")
    print(result.cluster_metrics)
    print("\n--- Términos principales por clúster ---")
    for cluster_id, terms in result.cluster_terms.items():
        count = int(
            (result.dataframe[clustering_config.cluster_col_name] == cluster_id).sum()
        )
        print(f"Clúster {cluster_id} ({count} vinos): {', '.join(terms)}")


if __name__ == "__main__":
    main()
