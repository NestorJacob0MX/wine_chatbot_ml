"""Compatibility facade for the original data_layer module."""

import logging
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_layer import (  # noqa: E402
    CsvWineDataLoader,
    DataCleaner,
    DataLoader,
    StepTimer,
    WineCleaningConfig,
    WineDataCleaner,
)

logger = logging.getLogger("WinePipeline.DataLayer")

__all__ = [
    "CsvWineDataLoader",
    "DataCleaner",
    "DataLoader",
    "StepTimer",
    "WineCleaningConfig",
    "WineDataCleaner",
]


def main() -> None:
    csv_path = Path("../dataset/wines.csv")
    config = WineCleaningConfig()

    loader = CsvWineDataLoader(filepath=csv_path, columns=config.active_columns)
    cleaner = WineDataCleaner(config=config)

    try:
        with StepTimer("Carga de datos"):
            raw_df = loader.load()

        with StepTimer("Limpieza y validación de datos"):
            clean_df = cleaner.clean(raw_df)

        print("\n--- Resumen de limpieza ---")
        print(cleaner.last_report)
        print("\n--- Vista previa limpia ---")
        print(clean_df.head(3))

    except (FileNotFoundError, ValueError, KeyError, TypeError, RuntimeError) as err:
        logger.critical("Ejecución detenida por error controlado: %s", err)


if __name__ == "__main__":
    main()
