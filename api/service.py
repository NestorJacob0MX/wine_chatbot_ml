from dataclasses import dataclass, field
import logging
from threading import RLock
from typing import Any, Dict
from uuid import uuid4

from chatbot_layer import SommelierWineChatbot
from pipeline.orchestrator import WineChatbotPipeline

logger = logging.getLogger("WinePipeline.API")


@dataclass
class ChatSession:
    chatbot: SommelierWineChatbot
    lock: Any = field(default_factory=RLock)
    closed: bool = False


class ChatService:
    """Gestiona sesiones conversacionales que comparten un pipeline ya entrenado."""

    def __init__(self, pipeline: WineChatbotPipeline) -> None:
        self._pipeline = pipeline
        self._sessions: Dict[str, ChatSession] = {}
        self._sessions_lock = RLock()

    def create_session(self) -> str:
        session_id = str(uuid4())
        session = ChatSession(chatbot=self._pipeline.create_chatbot())
        with self._sessions_lock:
            self._sessions[session_id] = session
        logger.info("Sesión conversacional creada: %s", session_id)
        return session_id

    def send_message(self, session_id: str, message: str) -> tuple[str, bool, str]:
        with self._sessions_lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError("No existe la sesión solicitada.")

        with session.lock:
            if session.closed:
                raise RuntimeError("La conversación ya terminó. Inicia una nueva sesión.")

            closing = session.chatbot.is_goodbye_message(message)
            response = session.chatbot.reply(message)
            session.closed = closing
            language = session.chatbot.context.language
            currency = "MXN (tipo de cambio fijo: 1 USD = 20 MXN)" if language == "es" else "USD"
            closed = session.closed

        if closed:
            with self._sessions_lock:
                self._sessions.pop(session_id, None)
        return response, closed, currency

    def close_session(self, session_id: str) -> bool:
        with self._sessions_lock:
            return self._sessions.pop(session_id, None) is not None
