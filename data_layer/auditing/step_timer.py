import logging
import time
from typing import Optional

logger = logging.getLogger("WinePipeline.DataLayer")


class StepTimer:
    """Context Manager para medir tiempos y registrar errores de cada paso."""

    def __init__(self, step_name: str) -> None:
        self.step_name = step_name
        self._start_time: Optional[float] = None

    def __enter__(self) -> "StepTimer":
        self._start_time = time.perf_counter()
        logger.info("Inicio del paso: %s", self.step_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        elapsed = time.perf_counter() - (self._start_time or time.perf_counter())
        if exc_type is not None:
            logger.error(
                "Error en '%s' tras %.2fs: %s",
                self.step_name,
                elapsed,
                exc_val,
            )
            return False
        logger.info("Finalizado '%s' en %.2fs", self.step_name, elapsed)
        return True
