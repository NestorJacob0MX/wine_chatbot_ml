"""Compatibility facade for the original NLP and clustering module."""

from model_layer import (
    UnsupervisedClusterer,
    WineClusteringConfig,
    WineMiniBatchKMeansClusterer,
)

from . import TextVectorizer, TfidfWineVectorizer

__all__ = [
    "TextVectorizer",
    "TfidfWineVectorizer",
    "UnsupervisedClusterer",
    "WineClusteringConfig",
    "WineMiniBatchKMeansClusterer",
]
