import logging
import re
from typing import List, Optional

from recommendation_layer import WineRecommendation, WineRecommender

from ..config.chatbot_config import ChatbotConfig
from ..contracts.query_normalizer import QueryNormalizer
from ..contracts.wine_chatbot import WineChatbot
from ..models.chat_turn import ChatTurn
from ..models.conversation_context import ConversationContext
from ..normalizers.keyword_bilingual_normalizer import KeywordBilingualNormalizer

logger = logging.getLogger("WinePipeline.ChatbotLayer")


class SommelierWineChatbot(WineChatbot):
    """Chatbot conversacional y amigable que recomienda vinos para cenas y gustos."""

    _GREETING_PATTERNS = re.compile(
        r"^\s*(hola|buenas|buenos dias|buenos días|buenas tardes|buenas noches|hello|hi|hey)\b",
        re.IGNORECASE,
    )
    _PRICE_PATTERNS = [
        re.compile(
            r"(?:menos de|maximo|máximo|hasta|presupuesto de|under|max)\s*\$?\s*(\d+(?:\.\d+)?)",
            re.IGNORECASE,
        ),
        re.compile(r"\$\s*(\d+(?:\.\d+)?)\s*(?:o menos|max|máximo)?", re.IGNORECASE),
    ]

    def __init__(
        self,
        recommender: WineRecommender,
        config: Optional[ChatbotConfig] = None,
        normalizer: Optional[QueryNormalizer] = None,
    ) -> None:
        if recommender is None:
            raise ValueError("Se requiere una instancia válida de WineRecommender.")
        self._recommender = recommender
        self._config = config or ChatbotConfig()
        self._normalizer = normalizer or KeywordBilingualNormalizer()
        self._history: List[ChatTurn] = []
        self._context = ConversationContext(language=self._config.default_language)

    @property
    def history(self) -> List[ChatTurn]:
        """Retorna una copia superficial del historial de conversación."""
        return list(self._history)

    @property
    def context(self) -> ConversationContext:
        """Retorna el contexto actual de la conversación."""
        return self._context

    def reset_conversation(self) -> None:
        """Limpia el historial y reinicia el contexto de la sesión."""
        self._history.clear()
        self._context = ConversationContext(language=self._config.default_language)
        logger.info("Conversación reiniciada.")

    def _detect_language(self, text: str) -> str:
        spanish_markers = re.findall(
            r"\b(hola|cena|cenar|pollo|carne|vino|tinto|blanco|quiero|recomienda|pescado|pasta|queso|pesos|dolares|dólares)\b|[ñáéíóú¿¡]",
            text.lower(),
        )
        return "es" if spanish_markers else "en"

    def _extract_max_price(self, text: str) -> Optional[float]:
        for pattern in self._PRICE_PATTERNS:
            match = pattern.search(text)
            if match:
                try:
                    value = float(match.group(1))
                    if value > 0:
                        return value
                except ValueError:
                    continue
        return None

    def _is_greeting_only(self, text: str) -> bool:
        stripped = text.strip()
        return bool(self._GREETING_PATTERNS.match(stripped)) and len(stripped.split()) <= 4

    def _format_greeting(self, lang: str) -> str:
        if lang == "en":
            return (
                f"Hi! I'm **{self._config.bot_name}**, your wine pairing assistant. "
                "Tell me what you're having for dinner or what flavors you enjoy (for example: "
                "*grilled salmon with lemon* or *a bold red for steak*), and I'll suggest great wines!"
            )
        return (
            f"¡Hola! Soy **{self._config.bot_name}**, tu asistente de vinos. "
            "Cuéntame qué vas a cenar o qué sabores te gustaría probar (por ejemplo: "
            "*salmón al horno con limón* o *un tinto con cuerpo para carne*) y te recomiendo opciones ideales."
        )

    def _format_recommendations(
        self,
        user_query: str,
        recs: List[WineRecommendation],
        lang: str,
    ) -> str:
        if not recs:
            if lang == "en":
                return (
                    "I couldn't find matching wines right now. Could you share a bit more about "
                    "your meal, flavors, or budget?"
                )
            return (
                "Por ahora no encontré coincidencias claras. ¿Me cuentas un poco más sobre tu cena, "
                "los sabores que buscas o tu presupuesto?"
            )

        if lang == "en":
            lines = [
                f"For **{user_query}**, here are some wine ideas you might really enjoy:\n"
            ]
        else:
            lines = [
                f"Para **{user_query}**, estas opciones encajan muy bien con lo que buscas:\n"
            ]

        for idx, rec in enumerate(recs, start=1):
            style_hint = self._config.cluster_hints.get(rec.cluster_id)
            reason_parts: List[str] = []
            if style_hint:
                reason_parts.append(
                    f"perfil de {style_hint}" if lang == "es" else f"{style_hint} profile"
                )
            if rec.points > 0:
                reason_parts.append(
                    f"calificación de {rec.points:.0f} pts"
                    if lang == "es"
                    else f"rated {rec.points:.0f} pts"
                )
            if rec.price > 0:
                reason_parts.append(
                    f"precio aprox. ${rec.price:.0f}"
                    if lang == "es"
                    else f"approx. ${rec.price:.0f}"
                )

            details = " · ".join(reason_parts)
            snippet = rec.description[:140].rstrip() + (
                "..." if len(rec.description) > 140 else ""
            )

            lines.append(f"**{idx}. {rec.title}** ({rec.variety})")
            if details:
                lines.append(f"   - {details}")
            if snippet:
                lines.append(
                    f"   - *Nota de cata:* {snippet}"
                    if lang == "es"
                    else f"   - *Tasting note:* {snippet}"
                )

        if lang == "en":
            lines.append(
                "\nWould you like something lighter, bolder, or within a specific price range?"
            )
        else:
            lines.append(
                "\n¿Prefieres algo más ligero, con más cuerpo o ajustado a algún presupuesto en especial?"
            )

        return "\n".join(lines)

    def reply(self, user_message: str) -> str:
        if (
            not isinstance(user_message, str)
            or len(user_message.strip()) < self._config.min_query_length
        ):
            logger.warning("Mensaje del usuario vacío o demasiado corto.")
            return (
                "Cuéntame un poco más sobre tu cena o lo que te gustaría tomar "
                "para poder recomendarte un vino."
            )

        clean_msg = user_message.strip()
        lang = self._detect_language(clean_msg)
        self._context.language = lang

        extracted_price = self._extract_max_price(clean_msg)
        if extracted_price is not None:
            self._context.max_price = extracted_price

        self._history.append(ChatTurn(role="user", message=clean_msg))

        if self._is_greeting_only(clean_msg):
            response_text = self._format_greeting(lang)
            self._history.append(ChatTurn(role="assistant", message=response_text))
            return response_text

        self._context.last_dinner_query = clean_msg
        search_query = self._normalizer.normalize_for_search(clean_msg, lang)

        try:
            recommendations = self._recommender.recommend(
                query_text=search_query,
                top_k=self._config.default_top_k,
                max_price=self._context.max_price,
            )
        except Exception as exc:
            logger.error("Error al generar recomendaciones en el chatbot: %s", exc)
            response_text = (
                "Tuve un inconveniente al buscar opciones de vino en este momento. "
                "¿Intentamos describir tu cena de otra forma?"
                if lang == "es"
                else "I ran into an issue looking up wine options right now. Could you describe your meal another way?"
            )
            self._history.append(ChatTurn(role="assistant", message=response_text))
            return response_text

        response_text = self._format_recommendations(clean_msg, recommendations, lang)
        self._history.append(
            ChatTurn(
                role="assistant",
                message=response_text,
                recommendations=tuple(recommendations),
            )
        )
        return response_text
