from abc import ABC, abstractmethod


class QueryNormalizer(ABC):
    """Contrato para normalizar consultas al idioma del catálogo."""

    @abstractmethod
    def normalize_for_search(self, user_text: str, detected_lang: str) -> str:
        """Retorna una consulta apta para buscar en el espacio vectorial."""
