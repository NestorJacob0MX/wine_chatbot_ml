"""Compatibility facade for model-layer components."""

from . import (
    UnsupervisedClusterer,
    WineClusteringConfig,
    WineMiniBatchKMeansClusterer,
)

__all__ = [
    "UnsupervisedClusterer",
    "WineClusteringConfig",
    "WineMiniBatchKMeansClusterer",
]
