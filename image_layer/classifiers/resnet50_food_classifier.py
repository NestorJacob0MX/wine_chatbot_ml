from io import BytesIO
import logging
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError
import torch
from torch import nn
from torchvision import models, transforms

from ..contracts.food_classifier import FoodPrediction

logger = logging.getLogger("WinePipeline.ImageLayer")

IMAGE_SIZE = 224
MAX_IMAGE_PIXELS = 25_000_000


class ResNet50FoodClassifier:
    """Clasificador de alimentos ResNet50 con transferencia desde ImageNet."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | None = None,
    ) -> None:
        self._checkpoint_path = Path(checkpoint_path)
        if not self._checkpoint_path.is_file():
            raise FileNotFoundError(
                f"No existe el checkpoint del clasificador: {self._checkpoint_path}. "
                "Entrénalo con `python -m image_layer.training.train_food_classifier`."
            )

        self._device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        )
        try:
            checkpoint: dict[str, Any] = torch.load(
                self._checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"No se pudo cargar el checkpoint '{self._checkpoint_path}': {exc}"
            ) from exc

        self._classes = checkpoint.get("classes")
        if not isinstance(self._classes, list) or not self._classes:
            raise ValueError("El checkpoint no contiene la lista de clases del modelo.")

        self._model = models.resnet50(weights=None)
        self._model.fc = nn.Linear(self._model.fc.in_features, len(self._classes))
        try:
            self._model.load_state_dict(checkpoint["model_state_dict"])
        except (KeyError, RuntimeError) as exc:
            raise ValueError(f"El checkpoint no coincide con ResNet50: {exc}") from exc
        self._model.to(self._device)
        self._model.eval()
        self._transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        logger.info(
            "Clasificador ResNet50 cargado (%d clases) en %s",
            len(self._classes),
            self._device,
        )

    @property
    def classes(self) -> list[str]:
        return list(self._classes)

    def predict(self, image_bytes: bytes) -> FoodPrediction:
        if not image_bytes:
            raise ValueError("El archivo de imagen está vacío.")

        try:
            with Image.open(BytesIO(image_bytes)) as source:
                if source.width * source.height > MAX_IMAGE_PIXELS:
                    raise ValueError("La imagen excede el límite permitido de píxeles.")
                image = source.convert("RGB")
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
            raise ValueError("El archivo recibido no es una imagen válida.") from exc

        tensor = self._transform(image).unsqueeze(0).to(self._device)
        with torch.inference_mode():
            probabilities = torch.softmax(self._model(tensor), dim=1)[0]
            confidence, class_index = torch.max(probabilities, dim=0)

        return FoodPrediction(
            label=self._classes[int(class_index.item())],
            confidence=float(confidence.item()),
        )
