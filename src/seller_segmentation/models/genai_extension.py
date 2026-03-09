"""Extensión GenAI – Asesor de Estrategia Comercial con LangChain + Anthropic.

Dado el cluster asignado a un vendedor y sus métricas operativas, una cadena LCEL
(LangChain Expression Language) genera diagnósticos e insights comerciales personalizados,
orientados al segmento al que pertenece el vendedor.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-sonnet-4-6"
_DEFAULT_PROFILES_PATH = Path(__file__).parents[3] / "data" / "cluster_profiles.md"

# Nombres estratégicos de cada segmento
CLUSTER_NAMES: dict[int, str] = {
    0: "Exploradores",
    1: "Exclusivos",
    2: "Consolidados",
    3: "Prometedores",
}

_SYSTEM_PROMPT = """
Eres un especialista en estrategia comercial con experiencia en marketplaces. Tu rol es
acompañar a vendedores de MercadoLibre en el diseño de estrategias de crecimiento adaptadas
a su perfil y segmento. Tu análisis parte siempre de los datos reales del vendedor: reconoces
sus fortalezas actuales y las usas como punto de partida para proponer iniciativas de alto
impacto.

Tu tono es profesional, directo y orientado al negocio. Escribe en español, en segunda persona
singular al dirigirte al vendedor. Cada recomendación debe ser específica, accionable y
justificada con base en los datos disponibles. Usa viñetas y no superes las 300 palabras.
""".strip()

_HUMAN_TEMPLATE = """

## Perfiles de segmentos de vendedores en MercadoLibre

{cluster_profiles}

---

## Datos del vendedor a analizar

- **Seller_Nickname:** {seller_id}
- **Segmento:** {cluster_name} (Cluster {cluster})
- **Categorías principales:** {categories}
- **Stock total disponible:** {stock}
- **Precio promedio:** {price}
- **Reputación:** {reputation}

---

Con base en el perfil del segmento y los datos del vendedor, elabora una ficha de estrategia
comercial personalizada con las siguientes secciones:

1. **Diagnóstico del negocio** — identifica 1 ventaja competitiva real del vendedor que puede
   aprovecharse como base para crecer dentro de su segmento.
2. **Estrategia de crecimiento** — propón 2 iniciativas comerciales concretas, ordenadas por
   prioridad, que el vendedor pueda implementar para aumentar sus ventas en el corto plazo.
3. **Indicador clave de éxito** — define el métrca principal que permitirá evaluar el impacto
   de la estrategia y hacer seguimiento del progreso.

""".strip()


# Reputación (1 = mejor, 9 = sin historial)
REPUTATION_MAP: dict[str, tuple[int, str]] = {
    "green_platinum": (1, "Platinum"),
    "green_gold": (2, "Oro"),
    "green_silver": (3, "Plata"),
    "green": (4, "Verde"),
    "light_green": (5, "Verde Claro"),
    "yellow": (6, "Amarillo"),
    "orange": (7, "Naranja"),
    "red": (8, "Rojo"),
    "newbie": (9, "Nuevo – sin historial suficiente"),
}


# ---------------------------------------------------------------------------
# Helpers de formato
# ---------------------------------------------------------------------------


def _format_reputation(raw: str) -> str:
    """Devuelve la reputación como etiqueta legible con su nivel ordinal."""
    entry = REPUTATION_MAP.get(raw.strip().lower())
    if entry is None:
        return raw or "No disponible"
    order, label = entry
    if order == 9:
        return f"{label} (nivel {order})"
    return f"{label} (nivel {order} de 8)"


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------


class StrategyAdvisor:
    """Genera diagnósticos e insights comerciales con LangChain + Anthropic.

    La cadena LCEL construye el prompt con el perfil del cluster (cargado desde
    un archivo Markdown) y los datos operativos del vendedor, y devuelve la
    estrategia generada por el LLM.

    Args:
        model: ID del modelo Anthropic a utilizar.
        profiles_path: Ruta al archivo Markdown con los perfiles de clusters.
            Por defecto apunta a ``data/cluster_profiles.md``.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        profiles_path: Path | str | None = None,
    ) -> None:
        self.model = model
        self._profiles_path = Path(profiles_path) if profiles_path else _DEFAULT_PROFILES_PATH
        self._cluster_profiles = self._load_profiles()
        self._chain = self._build_chain()

    # ------------------------------------------------------------------
    # Construcción de la cadena
    # ------------------------------------------------------------------

    def _load_profiles(self) -> str:
        with open(self._profiles_path, encoding="utf-8") as f:
            return f.read()

    def _build_chain(self) -> Runnable:
        """Construye la cadena LCEL: prompt | LLM | parser."""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM_PROMPT),
                ("human", _HUMAN_TEMPLATE),
            ]
        )
        llm = ChatAnthropic(model_name=self.model, timeout=None, stop=None)
        return prompt | llm | StrOutputParser()

    # ------------------------------------------------------------------
    # Formateo de datos del vendedor
    # ------------------------------------------------------------------

    def _build_input(
        self,
        seller_id: str | int,
        cluster: int,
        metrics: pd.Series,
    ) -> dict:
        stock = float(metrics.get("stock", 0))
        price = float(metrics.get("price", 0))
        reputation_raw = str(metrics.get("reputacion", ""))
        categories = str(metrics.get("categorias_seller", "")) or "No disponible"

        return {
            "cluster_profiles": self._cluster_profiles,
            "seller_id": seller_id,
            "cluster": cluster,
            "cluster_name": CLUSTER_NAMES.get(cluster, f"Cluster {cluster}"),
            "categories": categories,
            "stock": f"{stock:.0f} unidades",
            "price": f"${price:,.2f}",
            "reputation": _format_reputation(reputation_raw),
        }

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    @property
    def prompt_template(self) -> ChatPromptTemplate:
        """Devuelve el prompt template para inspección."""
        return self._chain.steps[0]  # type: ignore[attr-defined]

    def advise(
        self,
        seller_id: str | int,
        cluster: int,
        metrics: pd.Series,
    ) -> str:
        """Genera una estrategia comercial personalizada para un vendedor.

        Args:
            seller_id: Identificador o nickname del vendedor.
            cluster: Etiqueta del cluster asignado.
            metrics: Serie con las features del vendedor
                (stock, price, categorias_seller, reputacion).

        Returns:
            Texto de estrategia generado por el LLM.
        """
        input_data = self._build_input(seller_id, cluster, metrics)
        strategy = self._chain.invoke(input_data)
        logger.info(
            "Estrategia generada para %s (cluster %d – %s).",
            seller_id,
            cluster,
            CLUSTER_NAMES.get(cluster, "?"),
        )
        return strategy

    def advise_batch(
        self,
        df_perfil: pd.DataFrame,
        n: int = 1,
        seller_col: str = "seller_nickname",
        cluster_col: str = "cluster",
    ) -> pd.DataFrame:
        """Genera estrategias para una muestra de vendedores por cluster.

        Args:
            df_perfil: DataFrame a nivel seller con columnas seller_nickname,
                stock, price, categorias_seller, reputacion, cluster, cluster_name.
            n: Número de vendedores a procesar por cluster.
            seller_col: Nombre de la columna del seller.
            cluster_col: Nombre de la columna de cluster.

        Returns:
            DataFrame con columnas [seller_id, cluster, cluster_name, strategy].
        """
        records = []
        for cluster_id in sorted(df_perfil[cluster_col].dropna().unique()):
            muestra = df_perfil[df_perfil[cluster_col] == cluster_id].head(n)
            for _, row in muestra.iterrows():
                sid = row[seller_col]
                strategy = self.advise(sid, int(cluster_id), row)
                records.append(
                    {
                        "seller_id": sid,
                        "cluster": cluster_id,
                        "cluster_name": CLUSTER_NAMES.get(int(cluster_id), ""),
                        "strategy": strategy,
                    }
                )
        return pd.DataFrame(records)
