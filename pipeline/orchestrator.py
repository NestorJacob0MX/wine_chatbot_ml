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
from recommendation_layer import (
    ContentClusterWineRecommender,
    RecommendationConfig,
    WineRecommendation,
)
from chatbot_layer import ChatTurn, ChatbotConfig, SommelierWineChatbot


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
        recommendation_config: RecommendationConfig | None = None,
        chatbot_config: ChatbotConfig | None = None,
    ) -> None:
        self._data_loader = data_loader
        self._data_cleaner = data_cleaner
        self._text_vectorizer = text_vectorizer
        self._clusterer = clusterer
        self._cluster_col_name = cluster_col_name
        self._recommendation_config = recommendation_config or RecommendationConfig(
            cluster_col=cluster_col_name
        )
        self._chatbot_config = chatbot_config
        self._recommender: ContentClusterWineRecommender | None = None
        self._chatbot: SommelierWineChatbot | None = None

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
        self._recommender = ContentClusterWineRecommender(
            catalog_df=result_df,
            item_features=features,
            vectorizer=self._text_vectorizer,
            cluster_terms=cluster_terms,
            config=self._recommendation_config,
        )
        self._chatbot = SommelierWineChatbot(
            recommender=self._recommender,
            config=self._chatbot_config,
        )

        return WinePipelineResult(
            dataframe=result_df,
            feature_matrix=features,
            cluster_terms=cluster_terms,
            cluster_metrics=self._clusterer.metrics,
        )

    def recommend(
        self,
        query_text: str,
        top_k: int | None = None,
        max_price: float | None = None,
    ) -> List[WineRecommendation]:
        """Recomienda vinos para una consulta tras ejecutar el pipeline."""
        if self._recommender is None:
            raise RuntimeError("Primero debe ejecutarse run() para preparar el recomendador.")
        return self._recommender.recommend(
            query_text=query_text,
            top_k=top_k,
            max_price=max_price,
        )

    def reply(self, user_message: str) -> str:
        """Procesa un mensaje conversacional después de inicializar el pipeline."""
        if self._chatbot is None:
            raise RuntimeError("Primero debe ejecutarse run() para iniciar el chatbot.")
        return self._chatbot.reply(user_message)

    def is_goodbye_message(self, user_message: str) -> bool:
        """Indica si el mensaje debe terminar el bucle de conversación."""
        if self._chatbot is None:
            raise RuntimeError("Primero debe ejecutarse run() para iniciar el chatbot.")
        return self._chatbot.is_goodbye_message(user_message)

    def farewell_response(self, user_message: str = "") -> str:
        """Retorna una despedida adecuada al idioma de la conversación."""
        if self._chatbot is None:
            raise RuntimeError("Primero debe ejecutarse run() para iniciar el chatbot.")
        return self._chatbot.farewell_response(user_message)

    def reset_conversation(self) -> None:
        """Reinicia el historial y contexto conversacional actual."""
        if self._chatbot is None:
            raise RuntimeError("Primero debe ejecutarse run() para iniciar el chatbot.")
        self._chatbot.reset_conversation()

    @property
    def conversation_history(self) -> List[ChatTurn]:
        """Retorna el historial de la conversación actual."""
        if self._chatbot is None:
            raise RuntimeError("Primero debe ejecutarse run() para iniciar el chatbot.")
        return self._chatbot.history


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

    print("\nEscribe tu cena o preferencia (o 'salir' para terminar):")
    while True:
        user_message = input("\nTú: ").strip()
        if not user_message:
            print("SommelierAI: Cuéntame qué vas a cenar o qué tipo de vino prefieres.")
            continue
        if pipeline.is_goodbye_message(user_message):
            print(f"SommelierAI: {pipeline.farewell_response(user_message)}")
            break
        print(f"\nSommelierAI:\n{pipeline.reply(user_message)}")


if __name__ == "__main__":
    main()
