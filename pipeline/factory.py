import os
from pathlib import Path

from data_layer import CsvWineDataLoader, WineCleaningConfig, WineDataCleaner
from model_layer import WineClusteringConfig, WineMiniBatchKMeansClusterer
from nlp_layer import TfidfWineVectorizer

from .orchestrator import WineChatbotPipeline


def create_pipeline() -> WineChatbotPipeline:
    """Construye el pipeline de producción usando el catálogo del proyecto."""
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "dataset" / "wines.csv"
    image_checkpoint_path = Path(
        os.getenv(
            "FOOD_CLASSIFIER_CHECKPOINT",
            str(project_root / "artifacts" / "food_classifier_resnet50.pt"),
        )
    )
    cleaning_config = WineCleaningConfig()
    clustering_config = WineClusteringConfig()

    return WineChatbotPipeline(
        data_loader=CsvWineDataLoader(
            filepath=dataset_path,
            columns=cleaning_config.active_columns,
        ),
        data_cleaner=WineDataCleaner(config=cleaning_config),
        text_vectorizer=TfidfWineVectorizer(config=clustering_config),
        clusterer=WineMiniBatchKMeansClusterer(config=clustering_config),
        cluster_col_name=clustering_config.cluster_col_name,
        image_classifier_checkpoint=image_checkpoint_path,
    )
