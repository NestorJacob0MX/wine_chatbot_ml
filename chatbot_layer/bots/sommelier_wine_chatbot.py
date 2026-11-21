import logging
import re
from datetime import datetime
import unicodedata
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
        r"menos|presupuesto|pesos|peso|mxn|degustar|degustación|degustacion|"
        r"catar|cata|probar|gracias|por favor|adios|adiós|buenas noches|"
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
    _MEAL_INTENT_MARKERS = re.compile(
        r"\b(comida|comer|almuerzo|cena|cenar|desayuno|acompañar|acompanar|"
        r"maridar|platillo|comida|meal|food|lunch|dinner|pair(?:ing)? with|"
        r"eat|eating)\b",
        re.IGNORECASE,
    )
    _TASTING_INTENT_MARKERS = re.compile(
        r"\b(degustar|degustación|degustacion|catar|cata|probar|por sí solo|"
        r"por si solo|solo|tasting|sip|on its own|by itself)\b",
        re.IGNORECASE,
    )
    _MEAL_DETAIL_MARKERS = re.compile(
        r"\b(ribeye|steak|beef|carne|res|pollo|chicken|pescado|fish|salmon|"
        r"salmón|shrimp|camarones|mariscos|seafood|pasta|pizza|queso|cheese|"
        r"cerdo|pork|cordero|lamb|ensalada|salad|arroz|rice|sushi|tacos|"
        r"hamburguesa|burger|costillas|ribs|pavo|turkey|verduras|vegetables)\b",
        re.IGNORECASE,
    )
    _MEAL_CUE_PATTERN = re.compile(
        r"\b(?:voy a (?:comer|cenar)|comer[eé]|cenar[eé]|"
        r"i(?:'m| am) having|i(?:'m| am) eating|having|eating)\s+"
        r"(.+?)(?:[,!?;.]|$)",
        re.IGNORECASE,
    )
    _MEAL_REQUEST_TAIL = re.compile(
        r"\b(?:qué vino|que vino|me recomiendas|recomiéndame|"
        r"recommend(?: me)?|what wine|which wine)\b.*$",
        re.IGNORECASE,
    )
    _FOOD_PAIRING_TERMS = {
        "arepas": "corn savory cheese medium-bodied red wine",
        "bánh_mì": "pork herbs spicy aromatic white wine light red",
        "burger": "grilled beef juicy full-bodied red wine tannins",
        "churros": "cinnamon sweet dessert wine",
        "crepes": "light fruit sparkling wine",
        "currywurst": "spicy sausage fruity red wine rose",
        "empanadas": "savory meat red wine",
        "falafel": "herbs spices crisp white wine rose",
        "fish_and_chips": "fried fish crisp white wine sparkling",
        "gelato": "sweet dessert wine sparkling",
        "hot_dog": "sausage light red wine rose",
        "kebab_(shish_kebab)": "grilled meat spices syrah red wine",
        "pad_thai": "spicy lime aromatic white wine riesling",
        "pani_puri": "spicy tangy sparkling wine riesling",
        "pizza_slice": "tomato acidity red wine sangiovese",
        "poutine": "savory rich full-bodied red wine",
        "pretzel": "salty sparkling wine white wine",
        "samosas": "spicy aromatic white wine off-dry",
        "shawarma": "spiced grilled meat syrah red wine",
        "tacos": "spicy meat salsa red wine rose",
    }
    _TASTING_STYLE_MARKERS = re.compile(
        r"\b(tinto|blanco|rosado|rosé|espumoso|champagne|dulce|seco|robusto|"
        r"robusta|ligero|ligera|suave|frutal|cabernet|merlot|pinot|syrah|"
        r"shiraz|chardonnay|sauvignon|riesling|red|white|rose|sparkling|"
        r"sweet|dry|bold|full[- ]bodied|light|smooth|fruity)\b",
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
            r"(?:menos de|maximo|máximo|hasta|presupuesto(?: es)?(?: de)?|under|max)\s*\$?\s*(\d+(?:\.\d+)?)",
            re.IGNORECASE,
        ),
        re.compile(r"\$\s*(\d+(?:\.\d+)?)\s*(?:o menos|max|máximo)?", re.IGNORECASE),
        re.compile(r"(\d+(?:\.\d+)?)\s*(?:pesos?|mxn)\b", re.IGNORECASE),
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

    def _daypart(self) -> str:
        hour = datetime.now().astimezone().hour
        return "dinner" if hour >= 18 or hour < 5 else "meal"

    def _detect_intent(self, text: str) -> Optional[str]:
        tasting_match = self._TASTING_INTENT_MARKERS.search(text)
        meal_match = self._MEAL_INTENT_MARKERS.search(text)
        if tasting_match and not meal_match:
            return "tasting"
        if meal_match and not tasting_match:
            return "meal"
        if tasting_match and meal_match:
            return "tasting" if tasting_match.start() < meal_match.start() else "meal"
        return None

    def _has_meal_details(self, text: str) -> bool:
        if self._MEAL_DETAIL_MARKERS.search(text):
            return True
        normalized = text.casefold().replace(" ", "_")
        return any(label.casefold() in normalized for label in self._FOOD_PAIRING_TERMS)

    def _extract_meal_description(self, text: str) -> Optional[str]:
        cue_match = self._MEAL_CUE_PATTERN.search(text)
        if cue_match:
            description = self._MEAL_REQUEST_TAIL.sub("", cue_match.group(1)).strip()
            if description:
                return description

        food_match = self._MEAL_DETAIL_MARKERS.search(text)
        return food_match.group(0) if food_match else None

    def _has_tasting_style(self, text: str) -> bool:
        return bool(self._TASTING_STYLE_MARKERS.search(text))

    def _display_price(self, price_usd: float, language: str) -> str:
        if language == "es":
            price_mxn = price_usd * 20
            return f"MXN ${price_mxn:,.0f}"
        return f"USD ${price_usd:,.2f}"

    def reply_with_food(self, food_label: str) -> str:
        """Genera una recomendación a partir del platillo predicho en una imagen."""
        normalized_label = unicodedata.normalize("NFC", food_label.strip())
        normalized_label = normalized_label.casefold().replace(" ", "_")
        if not normalized_label:
            raise ValueError("La etiqueta del alimento no puede estar vacía.")
        pairing_terms = self._FOOD_PAIRING_TERMS.get(normalized_label, "food wine pairing")
        friendly_label = normalized_label.replace("_", " ").replace("(", "").replace(")", "")

        self._context.intent = "meal"
        self._context.awaiting = None
        if self._context.language == "en":
            message = (
                f"I'm having {friendly_label}. Suggest a wine pairing with "
                f"these food characteristics: {pairing_terms}."
            )
        else:
            message = (
                f"Voy a comer {friendly_label}. Busco un vino para acompañarlo; "
                f"considera estos sabores y estilos: {pairing_terms}."
            )
        return self.reply(message)

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

        meal = (
            self._extract_meal_description(self._context.last_dinner_query or user_query)
            if self._context.intent == "meal"
            else None
        )
        opening = (
            self._dialogue.meal_recommendation_opening(meal, lang)
            if meal
            else self._dialogue.recommendation_opening(user_query, lang)
        )
        lines = [f"{opening}\n"]

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
                    f"precio aprox. {self._display_price(rec.price, lang)}"
                    if lang == "es"
                    else f"about {self._display_price(rec.price, lang)}"
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
            self._context.max_price = extracted_price / 20 if lang == "es" else extracted_price

        self._history.append(ChatTurn(role="user", message=clean_msg))

        if self._is_greeting_only(clean_msg):
            greeting = self._format_greeting(lang)
            self._context.awaiting = "intent"
            response_text = (
                f"{greeting}\n\n{self._dialogue.intent_question(lang)}"
            )
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
                "Of course. Thanks for stopping by; I'm here whenever you'd like help choosing a wine."
                if lang == "en"
                else "Claro, gracias por visitarme. Cuando quieras elegir un vino, aquí estaré para ayudarte."
            )
            self._history.append(ChatTurn(role="assistant", message=response_text))
            return response_text

        detected_intent = self._detect_intent(clean_msg)
        if detected_intent:
            self._context.intent = detected_intent
        elif self._context.intent is None:
            self._context.intent = self._detect_intent(
                self._context.last_dinner_query or ""
            )

        if self._context.awaiting == "intent" and self._context.intent is None:
            response_text = self._dialogue.intent_question(lang)
            self._history.append(ChatTurn(role="assistant", message=response_text))
            return response_text

        if self._context.intent == "meal" and not self._has_meal_details(clean_msg):
            self._context.awaiting = "meal"
            response_text = self._dialogue.meal_question(lang, self._daypart())
            self._history.append(ChatTurn(role="assistant", message=response_text))
            return response_text

        if self._context.intent == "tasting" and not self._has_tasting_style(clean_msg):
            self._context.awaiting = "tasting"
            response_text = self._dialogue.tasting_question(lang)
            self._history.append(ChatTurn(role="assistant", message=response_text))
            return response_text

        if self._context.awaiting in {"intent", "meal", "tasting"}:
            prefix = f"{self._dialogue.acknowledgment(lang)} "
        else:
            prefix = ""

        previous_query = self._context.last_dinner_query
        if (
            previous_query
            and self._REFINEMENT_MARKERS.search(clean_msg)
            and not self._has_meal_details(clean_msg)
        ):
            search_text = f"{previous_query} {clean_msg}"
        else:
            search_text = clean_msg
        self._context.last_dinner_query = search_text
        self._context.awaiting = None
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

        response_text = prefix + self._format_recommendations(clean_msg, recommendations, lang)
        self._history.append(
            ChatTurn(
                role="assistant",
                message=response_text,
                recommendations=tuple(recommendations),
            )
        )
        return response_text
