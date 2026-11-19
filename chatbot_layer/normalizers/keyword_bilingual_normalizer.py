import logging
import re
from typing import Dict, List

from ..contracts.query_normalizer import QueryNormalizer

logger = logging.getLogger("WinePipeline.ChatbotLayer")


class KeywordBilingualNormalizer(QueryNormalizer):
    """Traduce términos culinarios y de cata frecuentes de español a inglés."""

    _ES_TO_EN_TERMS: Dict[str, str] = {
        "cena": "dinner",
        "comida": "food",
        "pollo": "chicken",
        "carne": "beef steak",
        "res": "beef",
        "cerdo": "pork",
        "cordero": "lamb",
        "pescado": "fish",
        "salmon": "salmon",
        "salmón": "salmon",
        "camarones": "shrimp",
        "mariscos": "seafood",
        "pasta": "pasta",
        "queso": "cheese",
        "ensalada": "salad",
        "picante": "spicy",
        "cremoso": "creamy",
        "salsa": "sauce",
        "a la parrilla": "grilled",
        "al horno": "baked",
        "vino tinto": "red wine",
        "tinto": "red",
        "vino blanco": "white wine",
        "blanco": "white",
        "dulce": "sweet",
        "seco": "dry",
        "frutal": "fruity",
        "afrutado": "fruity",
        "citrico": "citrus",
        "cítrico": "citrus",
        "fresco": "fresh crisp",
        "ligero": "light",
        "cuerpo": "body",
        "roble": "oak",
        "postre": "dessert",
    }

    def normalize_for_search(self, user_text: str, detected_lang: str) -> str:
        cleaned = user_text.strip()
        if detected_lang != "es":
            return cleaned

        lowered = cleaned.lower()
        translated_tokens: List[str] = [cleaned]

        for es_term, en_term in self._ES_TO_EN_TERMS.items():
            if re.search(rf"\b{re.escape(es_term)}\b", lowered):
                translated_tokens.append(en_term)

        normalized = " ".join(translated_tokens)
        logger.debug("Consulta normalizada ('%s') -> '%s'", cleaned, normalized)
        return normalized
