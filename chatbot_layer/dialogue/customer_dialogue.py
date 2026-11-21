from collections import defaultdict
from typing import DefaultDict, Dict, Tuple


class CustomerDialogue:
    """Genera respuestas amables con variación para mantener el diálogo natural."""

    _GREETINGS: Dict[str, Tuple[str, ...]] = {
        "es": (
            "¡Hola! Te doy la bienvenida a **{bot_name}**. Cuéntame qué vas a cenar y con gusto buscamos un vino para la ocasión.",
            "¡Qué gusto saludarte! Soy **{bot_name}**. ¿Qué platillo o sabores tienes en mente?",
            "¡Bienvenido/a! En **{bot_name}** puedo ayudarte a encontrar un vino para tu comida. ¿Qué vas a disfrutar hoy?",
        ),
        "en": (
            "Hello, and welcome to **{bot_name}**. Tell me what you're having and I'll gladly help find a wine for the occasion.",
            "It's lovely to meet you! I'm **{bot_name}**. What dish or flavors do you have in mind?",
            "Welcome to **{bot_name}**. I'd be happy to help you find a wine for your meal. What are you enjoying today?",
        ),
    }
    _FAREWELLS: Dict[str, Tuple[str, ...]] = {
        "es": (
            "Gracias por conversar conmigo. ¡Que disfrutes mucho tu comida y que tengas una excelente noche!",
            "Ha sido un gusto ayudarte. ¡Salud y que disfrutes tu vino!",
            "Gracias por visitar **{bot_name}**. Cuando quieras otra recomendación, aquí estaré. ¡Que tengas un lindo día!",
        ),
        "en": (
            "Thanks for chatting with me. Enjoy your meal, and have a lovely evening!",
            "It was a pleasure helping you. Cheers, and enjoy your wine!",
            "Thanks for visiting **{bot_name}**. I'll be here whenever you'd like another recommendation. Take care!",
        ),
    }
    _RECOMMENDATION_OPENINGS: Dict[str, Tuple[str, ...]] = {
        "es": (
            "Para **{query}**, encontré algunas opciones que podrían quedar muy bien:",
            "Pensando en **{query}**, estas alternativas podrían acompañar muy bien tu comida:",
            "¡Buena elección! Para **{query}**, consideraría estas opciones:",
            "Con **{query}** en mente, te propongo estos vinos:",
        ),
        "en": (
            "For **{query}**, I found a few wines that could pair nicely:",
            "Thinking about **{query}**, these options may be lovely with your meal:",
            "That sounds delicious! For **{query}**, I'd consider these wines:",
            "With **{query}** in mind, here are a few suggestions:",
        ),
    }
    _MEAL_RECOMMENDATION_OPENINGS: Dict[str, Tuple[str, ...]] = {
        "es": (
            "Veo que disfrutarás **{meal}**; ¡qué rico! Estos vinos pueden acompañarlo de maravilla:",
            "¡Qué buena elección, **{meal}**! Para ese platillo, te recomiendo estas opciones:",
            "Me encanta la idea de **{meal}**. Busquemos un vino que realce sus sabores:",
        ),
        "en": (
            "That sounds delicious—**{meal}**! Here are a few wines that could make a lovely pairing:",
            "I see you're having **{meal}**—that sounds lovely! These wines could pair beautifully:",
            "Great choice with **{meal}**! Here are a few pairings to consider:",
        ),
    }
    _RECOMMENDATION_CLOSINGS: Dict[str, Tuple[str, ...]] = {
        "es": (
            "Si me cuentas qué estilo prefieres, puedo afinar la selección.",
            "También puedo buscar alternativas por variedad, perfil o presupuesto.",
            "Ojalá alguna de estas opciones te guste; si quieres, seguimos explorando otros estilos.",
            "Cuando quieras, puedo proponerte opciones más ligeras, con más cuerpo o de otro precio.",
            "Espero que encuentres una buena compañía para tu comida. ¿Te gustaría explorar otro estilo?",
        ),
        "en": (
            "If you tell me which style you prefer, I can narrow down the selection.",
            "I can also look for alternatives by grape, flavor profile, or budget.",
            "I hope one of these catches your eye; we can explore other styles whenever you'd like.",
            "I can suggest lighter, fuller-bodied, or differently priced options whenever you're ready.",
            "I hope you find a lovely pairing. Would you like to explore another style?",
        ),
    }
    _NO_MATCH_MESSAGES: Dict[str, Tuple[str, ...]] = {
        "es": (
            "No encontré una coincidencia cercana esta vez. Si me das algún detalle más del platillo o del presupuesto, lo intento de nuevo.",
            "Aún no doy con el vino ideal. ¿Hay algún sabor, variedad o rango de precio que te gustaría priorizar?",
            "No apareció una opción clara para esa búsqueda. Podemos probar con otro ingrediente, estilo o presupuesto.",
        ),
        "en": (
            "I couldn't find a close match this time. A little more about the dish or your budget would help me try again.",
            "I haven't found the right wine yet. Is there a flavor, grape, or price range you'd like me to focus on?",
            "Nothing stood out for that search. We could try another ingredient, style, or budget.",
        ),
    }
    _INTENT_QUESTIONS: Dict[str, Tuple[str, ...]] = {
        "es": (
            "¿Buscas un vino para degustar por sí solo o para acompañar tu comida?",
            "¿Te gustaría elegir un vino para disfrutarlo solo, o buscas algo para acompañar tus alimentos?",
            "Para orientarte mejor, ¿quieres un vino para degustar o para maridar con tu comida?",
        ),
        "en": (
            "Are you looking for a wine to sip and enjoy on its own, or to pair with a meal?",
            "Would you like a wine for tasting on its own, or something to go with your food?",
            "To help you best, are you choosing a wine to enjoy by itself or with a meal?",
        ),
    }
    _MEAL_QUESTIONS: Dict[str, Dict[str, Tuple[str, ...]]] = {
        "es": {
            "meal": (
                "¡Con gusto! ¿Qué vas a comer? Cuéntame el platillo y, si quieres, también el presupuesto.",
                "Perfecto, busquemos un buen maridaje. ¿Qué comerás y hay algún presupuesto que quieras respetar?",
                "¡Claro! ¿Qué platillo tienes pensado para hoy? Con eso puedo sugerirte un vino que lo acompañe.",
            ),
            "dinner": (
                "¡Con gusto! ¿Qué vas a cenar? Cuéntame el platillo y, si quieres, también el presupuesto.",
                "Perfecto, busquemos un buen maridaje para esta noche. ¿Qué cenarás?",
                "¡Claro! ¿Qué tienes pensado para la cena? Así puedo recomendarte algo que combine bien.",
            ),
        },
        "en": {
            "meal": (
                "I'd be happy to help. What are you having to eat? You can also share a budget if you have one.",
                "Lovely. What dish are you planning to enjoy, and is there a budget you'd like me to keep in mind?",
                "Of course! Tell me what you're eating, and I'll find a wine that pairs nicely.",
            ),
            "dinner": (
                "I'd be happy to help. What are you having for dinner? You can also share a budget if you have one.",
                "Let's find a lovely pairing for tonight. What are you planning for dinner?",
                "Of course! What are you having for dinner? I'll suggest something that pairs well.",
            ),
        },
    }
    _TASTING_QUESTIONS: Dict[str, Tuple[str, ...]] = {
        "es": (
            "¡Qué buena idea! ¿Qué estilo o tipo de vino disfrutas: tinto, blanco, rosado, dulce, seco o con más cuerpo?",
            "Para degustarlo, ¿qué perfil te gusta más? Por ejemplo, un tinto robusto, un blanco fresco o algo dulce.",
            "¿Tienes alguna variedad o estilo favorito para degustar? También puedo sugerirte algo nuevo.",
        ),
        "en": (
            "Sounds lovely! What style do you enjoy: red, white, rosé, sweet, dry, or full-bodied?",
            "For a tasting, what kind of profile do you prefer? For example, a bold red, a crisp white, or something sweet.",
            "Do you have a favorite grape or style to sip, or would you like to explore something new?",
        ),
    }
    _ACKNOWLEDGMENTS: Dict[str, Tuple[str, ...]] = {
        "es": (
            "Entendido.",
            "Perfecto, gracias por contármelo.",
            "¡Muy bien, ya tengo una idea!",
        ),
        "en": (
            "Got it.",
            "Perfect, thanks for sharing that.",
            "Lovely, I have a good starting point.",
        ),
    }

    def __init__(self) -> None:
        self._positions: DefaultDict[str, int] = defaultdict(int)

    def _next(self, key: str, choices: Tuple[str, ...]) -> str:
        position = self._positions[key]
        self._positions[key] += 1
        return choices[position % len(choices)]

    def greeting(self, bot_name: str, language: str) -> str:
        template = self._next("greeting", self._GREETINGS[language])
        return template.format(bot_name=bot_name)

    def intent_question(self, language: str) -> str:
        return self._next("intent_question", self._INTENT_QUESTIONS[language])

    def meal_question(self, language: str, daypart: str) -> str:
        return self._next(
            f"meal_question_{daypart}",
            self._MEAL_QUESTIONS[language][daypart],
        )

    def tasting_question(self, language: str) -> str:
        return self._next("tasting_question", self._TASTING_QUESTIONS[language])

    def acknowledgment(self, language: str) -> str:
        return self._next("acknowledgment", self._ACKNOWLEDGMENTS[language])

    def farewell(self, bot_name: str, language: str) -> str:
        template = self._next("farewell", self._FAREWELLS[language])
        return template.format(bot_name=bot_name)

    def recommendation_opening(self, query: str, language: str) -> str:
        template = self._next(
            "recommendation_opening",
            self._RECOMMENDATION_OPENINGS[language],
        )
        return template.format(query=query)

    def meal_recommendation_opening(self, meal: str, language: str) -> str:
        template = self._next(
            "meal_recommendation_opening",
            self._MEAL_RECOMMENDATION_OPENINGS[language],
        )
        return template.format(meal=meal)

    def recommendation_closing(self, language: str) -> str:
        return self._next(
            "recommendation_closing",
            self._RECOMMENDATION_CLOSINGS[language],
        )

    def no_match(self, language: str) -> str:
        return self._next("no_match", self._NO_MATCH_MESSAGES[language])
