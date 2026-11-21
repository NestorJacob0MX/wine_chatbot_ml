import argparse
import json
import logging
from pathlib import Path
import random
from typing import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

logger = logging.getLogger("WinePipeline.ImageTraining")
IMAGE_SIZE = 224


def _stratified_indices(
    targets: Sequence[int],
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    class_indices: dict[int, list[int]] = {}
    for index, target in enumerate(targets):
        class_indices.setdefault(int(target), []).append(index)

    train_indices: list[int] = []
    validation_indices: list[int] = []
    for indices in class_indices.values():
        if len(indices) < 2:
            raise ValueError(
                "Cada clase necesita al menos dos imágenes para dividir "
                "entrenamiento y validación."
            )
        rng.shuffle(indices)
        validation_count = min(
            len(indices) - 1,
            max(1, round(len(indices) * validation_fraction)),
        )
        validation_indices.extend(indices[:validation_count])
        train_indices.extend(indices[validation_count:])

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return train_indices, validation_indices


def train(
    dataset_dir: Path,
    output_path: Path,
    epochs: int = 8,
    batch_size: int = 32,
    validation_fraction: float = 0.2,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> dict[str, object]:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"No se encontró el dataset: {dataset_dir}")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs y batch_size deben ser mayores que cero.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction debe estar entre cero y uno.")

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    validation_transform = transforms.Compose(
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

    train_dataset = datasets.ImageFolder(dataset_dir, transform=train_transform)
    validation_dataset = datasets.ImageFolder(
        dataset_dir,
        transform=validation_transform,
    )
    if len(train_dataset.classes) < 2:
        raise ValueError("El dataset debe contener al menos dos carpetas/clases.")

    train_indices, validation_indices = _stratified_indices(
        train_dataset.targets,
        validation_fraction,
        seed,
    )
    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        Subset(validation_dataset, validation_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    best_accuracy = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_count = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_count += labels.size(0)

        model.eval()
        validation_loss = 0.0
        validation_correct = 0
        validation_count = 0
        with torch.inference_mode():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                validation_loss += loss.item() * labels.size(0)
                validation_correct += (logits.argmax(dim=1) == labels).sum().item()
                validation_count += labels.size(0)

        validation_accuracy = validation_correct / validation_count
        epoch_metrics = {
            "train_loss": train_loss / train_count,
            "train_accuracy": train_correct / train_count,
            "validation_loss": validation_loss / validation_count,
            "validation_accuracy": validation_accuracy,
        }
        history.append(epoch_metrics)
        logger.info("Época %d/%d: %s", epoch + 1, epochs, epoch_metrics)

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("El entrenamiento terminó sin producir un modelo válido.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": "resnet50",
            "image_size": IMAGE_SIZE,
            "classes": train_dataset.classes,
            "model_state_dict": best_state,
            "validation_accuracy": best_accuracy,
        },
        output_path,
    )
    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "dataset": str(dataset_dir),
                "classes": train_dataset.classes,
                "train_images": len(train_indices),
                "validation_images": len(validation_indices),
                "best_validation_accuracy": best_accuracy,
                "history": history,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Mejor clasificador guardado en %s", output_path)
    return {"checkpoint": str(output_path), "validation_accuracy": best_accuracy}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Entrena ResNet50 para clasificar platillos.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=project_root / "dataset" / "dataset_food",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "artifacts" / "food_classifier_resnet50.pt",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(
        dataset_dir=args.dataset,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
