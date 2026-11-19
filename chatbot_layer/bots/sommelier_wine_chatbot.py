import logging
import re
from typing import List, Optional

from recommendation_layer import WineRecommendation, WineRecommender

from ..config.chatbot_config import ChatbotConfig
from ..contracts.query_normalizer import QueryNormalizer
from ..contracts.wine_chatbot import WineChatbot
from ..dialogue.customer_dialogue import CustomerDialogue
from ..models.chat_turn import ChatTurn
from ..models.conversation_context import ConversationContext
from ..normalizers.keyword_bilingual_normalizer import KeywordBilingualNormalizer

logger = logging.getLogger("WinePipeline.ChatbotLayer")


class SommelierWineChatbot(WineChatbot):
    """Chatbot conversacional y amigable que recomienda vinos para cenas y gustos."""

    _GREETING_PATTERNS = re.compile(
        r"^\s*(hola|buenas|buenos dias|buenos días|buenas tardes|buenas noches|"
        r"hello|hi|hey|good morning|good afternoon|good evening)\b",
        re.IGNORECASE,
    )
    _GOODBYE_PHRASE_PATTERN = re.compile(
        r"\b(salir|exit|quit|adios|hasta luego|hasta pronto|hasta mañana|"
        r"nos vemos|me voy|tengo que irme|bye|goodbye|good night|"
        r"see you|take care|that's it|that's all|that is all|i'm done|im done)\b",
        re.IGNORECASE,
    )
    _SPANISH_MARKERS = re.compile(
        r"\b(hola|cena|cenar|comer|comida|pollo|carne|vino|tinto|blanco|"
        r"quiero|quisiera|recomienda|recomiendas|pescado|pasta|queso|"
        r"necesito|requiero|algo|robusto|robusta|ligero|ligera|mas|más|"
        r"menos|presupuesto|gracias|por favor|adios|adiós|buenas noches|"
        r"hasta luego|"
        r"hasta pronto|hasta mañana|nos vemos|me voy|salir|"
        r"voy|comer|con|para|pure|puré|papas|patatas)\b|[ñáéíóú¿¡]",
        re.IGNORECASE,
    )
    _ENGLISH_MARKERS = re.compile(
        r"\b(hello|hi|hey|dinner|eat|food|chicken|beef|steak|wine|red|white|"
        r"want|would|recommend|fish|pasta|cheese|need|something|bold|"
        r"full bodied|light|lighter|bolder|more|less|budget|please|"
        r"thanks|thank you|goodbye|bye|exit|quit|with|for|i|"
        r"see you|take care|good morning|good afternoon|good evening|good night)\b",
        re.IGNORECASE,
    )
    _REFINEMENT_MARKERS = re.compile(
        r"\b(algo|más|mas|menos|robusto|robusta|ligero|ligera|suave|"
        r"frutal|económico|economico|barato|something|more|less|"
        r"bolder|lighter|softer|full bodied|bold|light|fruity|cheaper)\b",
        re.IGNORECASE,
    )
    _NEGATIVE_ACKNOWLEDGMENTS = {
        "no",
        "no gracias",
        "no, gracias",
        "not really",
        "no thanks",
        "no thank you",
    }
    _THANKS_PATTERN = re.compile(
        r"^\s*(gracias|muchas gracias|thanks|thank you|thanks a lot)[.! ]*$",
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
        dialogue: Optional[CustomerDialogue] = None,
    ) -> None:
        if recommender is None:
            raise ValueError("Se requiere una instancia válida de WineRecommender.")
        self._recommender = recommender
        self._config = config or ChatbotConfig()
        self._normalizer = normalizer or KeywordBilingualNormalizer()
        self._dialogue = dialogue or CustomerDialogue()
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

    def is_goodbye_message(self, user_message: str) -> bool:
        """Indica si el usuario expresó que desea terminar la conversación."""
        normalized = re.sub(
            r"[.,!?¡¿;:]+",
            " ",
            user_message.casefold(),
        )
        normalized = " ".join(normalized.split())
        if not normalized:
            return False

        if normalized in self._NEGATIVE_ACKNOWLEDGMENTS:
            return bool(self._history)

        if "buenas noches" in normalized:
            return bool(self._history)

        if self._GOODBYE_PHRASE_PATTERN.search(normalized):
            return True
        return False

    def farewell_response(self, user_message: str = "") -> str:
        """Genera una despedida en el idioma detectado o actual."""
        language = self._detect_language(user_message) if user_message.strip() else self._context.language
        return self._dialogue.farewell(self._config.bot_name, language)

    def _detect_language(self, text: str) -> str:
        spanish_matches = self._SPANISH_MARKERS.findall(text)
        english_matches = self._ENGLISH_MARKERS.findall(text)
        if len(spanish_matches) > len(english_matches):
            return "es"
        if len(english_matches) > len(spanish_matches):
            return "en"
        if spanish_matches:
            return self._context.language
        return self._context.language or self._config.default_language

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
        return self._dialogue.greeting(self._config.bot_name, lang)

    def _format_recommendations(
        self,
        user_query: str,
        recs: List[WineRecommendation],
        lang: str,
    ) -> str:
        if not recs:
            return self._dialogue.no_match(lang)

        lines = [
            f"{self._dialogue.recommendation_opening(user_query, lang)}\n"
        ]

        for idx, rec in enumerate(recs, start=1):
            hints = (
                self._config.cluster_hints
                if lang == "es"
                else self._config.cluster_hints_en
            )
            style_hint = hints.get(rec.cluster_id)
            reason_parts: List[str] = []
            if style_hint:
                reason_parts.append(f"perfil {style_hint}" if lang == "es" else style_hint)
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

        lines.append(f"\n{self._dialogue.recommendation_closing(lang)}")

        return "\n".join(lines)

    def reply(self, user_message: str) -> str:
        if (
            not isinstance(user_message, str)
            or not user_message.strip()
        ):
            logger.warning("Mensaje del usuario vacío o demasiado corto.")
            if self._context.language == "en":
                return "Please tell me a little about your meal or the kind of wine you enjoy."
            return "Cuéntame un poco sobre tu platillo o el tipo de vino que disfrutas."

        clean_msg = user_message.strip()
        if self.is_goodbye_message(clean_msg):
            response_text = self.farewell_response(clean_msg)
            self._history.append(ChatTurn(role="user", message=clean_msg))
            self._history.append(ChatTurn(role="assistant", message=response_text))
            return response_text

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

        if self._THANKS_PATTERN.fullmatch(clean_msg):
            response_text = (
                "You're very welcome. I'm here whenever you'd like another wine suggestion."
                if lang == "en"
                else "Con mucho gusto. Aquí estaré cuando quieras otra recomendación."
            )
            self._history.append(ChatTurn(role="assistant", message=response_text))
            return response_text

        if len(clean_msg) < self._config.min_query_length:
            response_text = (
                "Could you tell me a little more about your meal or wine preference?"
                if lang == "en"
                else "¿Me cuentas un poco más sobre tu platillo o el vino que prefieres?"
            )
            self._history.append(ChatTurn(role="assistant", message=response_text))
            return response_text

        if clean_msg.casefold() in self._NEGATIVE_ACKNOWLEDGMENTS:
            response_text = (
                "No problem. If you'd like another suggestion later, just tell me what you're having."
                if lang == "en"
                else "Claro, no hay problema. Si después quieres otra sugerencia, cuéntame qué vas a comer."
            )
            self._history.append(ChatTurn(role="assistant", message=response_text))
            return response_text

        previous_query = self._context.last_dinner_query
        if previous_query and self._REFINEMENT_MARKERS.search(clean_msg):
            search_text = f"{previous_query} {clean_msg}"
        else:
            search_text = clean_msg
        self._context.last_dinner_query = search_text
        search_query = self._normalizer.normalize_for_search(search_text, lang)

        try:
            recommendations = self._recommender.recommend(
                query_text=search_query,
                top_k=self._config.default_top_k,
                max_price=self._context.max_price,
            )
        except Exception as exc:
            logger.error("Error al generar recomendaciones en el chatbot: %s", exc)
            response_text = (
                "Lo siento, en este momento tuve un inconveniente al buscar opciones. "
                "¿Podrías intentarlo de nuevo o contarme un poco más sobre tu platillo?"
                if lang == "es"
                else "I'm sorry, I had trouble finding wine options just now. "
                "Would you mind trying again or telling me a little more about your meal?"
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
