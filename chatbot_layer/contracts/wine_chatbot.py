from abc import ABC, abstractmethod


class WineChatbot(ABC):
    """Contrato para agentes conversacionales de recomendación de vinos."""

    @abstractmethod
    def reply(self, user_message: str) -> str:
        """Procesa un mensaje y devuelve una respuesta conversacional."""
