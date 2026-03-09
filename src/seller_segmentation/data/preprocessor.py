"""Limpieza de datos y agregación a nivel de vendedor."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def clean_challenger_df(
    df: pd.DataFrame, lower_q: float = 0.25, upper_q: float = 0.95
) -> pd.DataFrame:
    """Aplica reglas básicas de limpieza a la tabla de challenge sin procesar.

    - Elimina filas sin identificador de seller.
    - Elimina duplicados obvios.
    - Elimina datos atípicos.
    - Trata valores nulls
    - Convierte columnas numéricas que llegaron como texto.

    Args:
        df: DataFrame de items sin procesar.

    Returns:
        DataFrame de items limpio.
    """

    df = df.copy()
    initial_len = len(df)

    # 1. Normalización de nombres de columnas
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)

    # 2. Conversión de tipos
    numeric_cols = ["stock", "price", "regular_price"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. Tratamiento de nulos
    if "seller_reputation" in df.columns:
        df["seller_reputation"] = df["seller_reputation"].fillna("sin_categoria")

    if "is_refurbished" in df.columns:
        df["is_refurbished"] = df["is_refurbished"].fillna("sin_categoria")

    if "condition" in df.columns:
        df["condition"] = df["condition"].fillna("sin_categoria")

    if "logistic_type" in df.columns:
        df["logistic_type"] = df["logistic_type"].fillna("sin_categoria")

    if "category_name" in df.columns:
        df["category_name"] = df["category_name"].fillna("sin_categoria")

    if "price" in df.columns:
        df["price"] = df["price"].fillna(df["price"].median())

    # regular_price se deja sin imputar por ahora

    # 4. Tratamiento de outliers (solo variables numéricas relevantes)
    outlier_cols = ["stock", "price", "regular_price"]

    for col in outlier_cols:
        if col in df.columns:
            lower = df[col].quantile(lower_q)
            upper = df[col].quantile(upper_q)
            df[col] = df[col].clip(lower, upper)

    # 5. Limpieza de variables categóricas
    categorical_cols = [
        "seller_nickname",
        "seller_reputation",
        "logistic_type",
        "condition",
        "category_name",
    ]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
                .str.replace(" ", "_")
                .str.lower()
            )
    # 6. Limpieza de duplicados y nulls
    df = df.dropna(subset=["seller_nickname"])
    df = df.drop_duplicates()

    dropped = initial_len - len(df)
    logger.info(
        "Eliminadas %d filas en la limpieza (%.1f%%).", dropped, 100 * dropped / initial_len
    )

    return df


def identificar_columnas_mono_categoria(
    df: pd.DataFrame, col_group: str, cols_categoricas: list[str]
) -> tuple[pd.DataFrame, dict]:
    """
    Identifica columnas categóricas que tienen un único valor por grupo.

    Args:
        df: DataFrame.
        col_group: columna por la cual agrupar.
        cols_categoricas: lista de columnas categóricas a evaluar.

    Returns:
        resumen_df: DataFrame resumen con métricas
        detalle: diccionario con sellers mono-categoría por variable
    """

    resumen = []
    detalle = {}

    total_grupos = df[col_group].nunique()

    for col in cols_categoricas:
        categorias_por_grupo = (
            df.groupby(col_group)[col].nunique().reset_index(name="num_categorias")
        )

        mono_categoria = categorias_por_grupo[categorias_por_grupo["num_categorias"] == 1]

        num_mono = len(mono_categoria)
        porcentaje = round(num_mono / total_grupos * 100, 2)

        resumen.append(
            {
                "columna": col,
                "total_grupos": total_grupos,
                "grupos_mono_categoria": num_mono,
                "porcentaje_mono_categoria": porcentaje,
            }
        )

        detalle[col] = mono_categoria

    resumen_df = pd.DataFrame(resumen)

    return resumen_df, detalle
