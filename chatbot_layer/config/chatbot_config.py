from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class ChatbotConfig:
    """Configuración del comportamiento conversacional del sommelier virtual."""

    bot_name: str = "SommelierAI"
    default_top_k: int = 3
    min_query_length: int = 2
    default_language: str = "es"
    cluster_hints: Dict[int, str] = field(
        default_factory=lambda: {
            0: "tintos con cuerpo, estructura y notas frutales/especiadas",
            1: "vinos frescos, frutales y con buena acidez",
            2: "blancos vibrantes con notas cítricas y de fruta fresca",
            3: "ensamblajes tintos elegantes estilo Burdeos",
            4: "tintos expresivos con fruta roja/negra y taninos definidos",
            5: "vinos suaves y aromáticos tipo Pinot Noir con frutos rojos",
        }
    )
    cluster_hints_en: Dict[int, str] = field(
        default_factory=lambda: {
            0: "structured, full-bodied reds with fruit and spice notes",
            1: "fresh, fruity wines with lively acidity",
            2: "vibrant whites with citrus and fresh fruit notes",
            3: "elegant Bordeaux-style red blends",
            4: "expressive reds with red or dark fruit and defined tannins",
            5: "smooth, aromatic Pinot Noir-style wines with red fruit",
        }
    )
