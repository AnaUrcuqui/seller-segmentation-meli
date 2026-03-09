"""Ingeniería de features para perfiles de sellers."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


def agregar_variables_numericas(
    df: pd.DataFrame,
    col_grupo: str,
    cols_numericas: list[str],
    agg_funcs: list[str] | None = None,
    nombre_conteo: str = "num_registros",
) -> pd.DataFrame:
    """
    Agrupa variables numéricas por una columna y renombra columnas.

    Args:
        - df: DataFrame con datos a nivel producto
        - col_grupo: Columna por la cual agrupar
        - cols_numericas: Lista de columnas numéricas a agregar
        - agg_funcs: Funciones de agregación a aplicar
        - nombre_conteo:Nombre para la columna de conteo

    Returns:
        DataFrame con columnas renombradas como nombre_funcion
    """

    if agg_funcs is None:
        agg_funcs = ["sum", "mean", "median", "std", "max"]
    agg_dict = {col: agg_funcs for col in cols_numericas}
    agg_dict[col_grupo] = ["count"]
    df_agrupado = df.groupby(col_grupo).agg(agg_dict).reset_index()
    df_agrupado.columns = ["_".join(col).strip("_") for col in df_agrupado.columns]

    # Limpiar nombre de columna de agrupación y conteo
    df_agrupado = df_agrupado.rename(
        columns={f"{col_grupo}_": col_grupo, f"{col_grupo}_count": nombre_conteo}
    )

    return df_agrupado


def transformar_variables_categoricas(
    df: pd.DataFrame, col_seller: str = "seller_nickname"
) -> pd.DataFrame:
    """
    Transforma variables categóricas de nivel producto a nivel seller según
    su comportamiento mono-categoría o multi-categoría.
        - is_refurbished: EXCLUIDA (baja variabilidad)
        - condition: One-hot encoding (2 categorías relevantes)
        - seller_reputation: Codificación ordinal (mantiene jerarquía)
        - logistic_type: One-hot encoding (5 categorías)
        - category_name: Métricas agregadas (alta cardinalidad - 54 categorías)

    Args:
        - df: DataFrame con datos a nivel producto
        - col_grupo: Nombre de la columna de seller


    Returns:
        DataFrame a nivel seller con variables transformadas
    """

    features_seller = pd.DataFrame(index=df[col_seller].unique())

    # ========================================================================
    # 1. CONDITION - One-Hot Encoding
    # ========================================================================
    # Crear dummies a nivel producto
    condition_dummies = pd.get_dummies(
        df[[col_seller, "condition"]], columns=["condition"], prefix="condition"
    )

    # Agregar a nivel seller
    condition_agg = condition_dummies.groupby(col_seller).sum()

    # Unir con features_seller
    features_seller = features_seller.join(condition_agg)

    # ========================================================================
    # 2. SELLER_REPUTATION - Codificación Ordinal
    # ========================================================================

    # Orden jerárquico de reputación (1 = mejor, 10 = sin categoría)
    reputation_order = {
        "green_platinum": 1,
        "green_gold": 2,
        "green_silver": 3,
        "green": 4,
        "light_green": 5,
        "yellow": 6,
        "orange": 7,
        "red": 8,
        "newbie": 9,
        "sin_categoria": 10,
    }

    # Obtener la reputación más común por seller (moda)
    seller_reputation = df.groupby(col_seller)["seller_reputation"].apply(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else None
    )

    # Codificar ordinalmente
    features_seller["seller_reputation_ordinal"] = seller_reputation.map(reputation_order)

    # ========================================================================
    # 3. LOGISTIC_TYPE - One-Hot Encoding
    # ========================================================================
    # Crear dummies a nivel producto
    logistic_dummies = pd.get_dummies(
        df[[col_seller, "logistic_type"]], columns=["logistic_type"], prefix="logistic"
    )

    # Agregar a nivel seller
    logistic_agg = logistic_dummies.groupby(col_seller).sum()

    # Unir con features_seller
    features_seller = features_seller.join(logistic_agg)

    # ========================================================================
    # 4. CATEGORY_NAME - Métricas Agregadas
    # ========================================================================

    # 4.1 Métricas de Diversidad del Portafolio
    # ------------------------------------------

    # Número de categorías únicas
    features_seller["num_categorias"] = df.groupby(col_seller)["category_name"].nunique()

    # Entropía de categorías (dispersión)
    features_seller["entropia_categorias"] = df.groupby(col_seller)["category_name"].apply(
        lambda x: entropy(x.value_counts(normalize=True))
    )

    # Concentración en categoría principal
    features_seller["concentracion"] = df.groupby(col_seller)["category_name"].apply(
        lambda x: (x.value_counts().iloc[0] / len(x)) if len(x) > 0 else 0
    )

    # 4.2 Participación de Principales Categorías en Valor
    # ------------------------------------------------------

    # Identificar top N categorías por frecuencia global
    top_n = 5
    top_categorias = df["category_name"].value_counts().nlargest(top_n).index

    # Calcular valor total por seller-categoría
    df_valor = df.groupby([col_seller, "category_name"]).size().unstack(fill_value=0)

    # Normalizar por seller (% del valor total en cada categoría)
    df_valor_pct = df_valor.div(df_valor.sum(axis=1), axis=0).fillna(0)

    # Agregar solo top categorías
    for cat in top_categorias:
        if cat in df_valor_pct.columns:
            safe_cat = cat.replace(" ", "_").replace("/", "_").replace("-", "_")[:30]
            col_name = f"pct_valor_{safe_cat}"
            features_seller[col_name] = df_valor_pct[cat]

    # Resetear índice
    features_seller = features_seller.fillna(0).reset_index()
    features_seller = features_seller.rename(columns={"index": "seller_nickname"})

    return features_seller


def validar_y_limpiar_variables_numericas(
    df: pd.DataFrame,
    col_id: str = "seller_nickname",
    umbral_completitud: float = 0.90,
    percentil_inferior: float = 0.25,
    percentil_superior: float = 0.95,
    umbral_variabilidad: float = 0.30,
) -> dict:
    """
    Valida y limpia variables numéricas a nivel seller aplicando criterios de:
    1. Completitud mínima (90%)
    2. Tratamiento de valores atípicos (winsorización P25-P95)
    3. Variabilidad mínima (30%)

    Args:
     - df : DataFrame a nivel seller con variables numéricas
     - col_id : Nombre de la columna identificadora del seller
     - umbral_completitud : % mínimo de valores poblados requerido, default 90%
     - percentil_inferior : Percentil inferior para tratamiento de atípicos, default=0.25
     - percentil_superior : Percentil superior para tratamiento de atípicos, default=0.95
     - umbral_variabilidad : % mínimo de variabilidad requerido, default=0.30

    Returns:
    dict con:
        - 'df_limpio': DataFrame procesado
        - 'reporte_completitud': Variables excluidas por completitud
        - 'reporte_atipicos': Estadísticas de tratamiento de atípicos
        - 'reporte_variabilidad': Variables excluidas por baja variabilidad
        - 'variables_excluidas': Lista total de variables excluidas
        - 'variables_finales': Lista de variables que pasaron validación
    """

    df_procesado = df.copy()

    # Identificar columnas numéricas (excluir ID)
    cols_numericas = df_procesado.select_dtypes(include=[np.number]).columns.tolist()
    if col_id in cols_numericas:
        cols_numericas.remove(col_id)

    # ========================================================================
    # ETAPA 1: VALIDACIÓN DE COMPLETITUD (>= 90%)
    # ========================================================================

    reporte_completitud = []
    variables_excluidas_completitud = []

    for col in cols_numericas:
        total_registros = len(df_procesado)
        valores_poblados = df_procesado[col].notna().sum()
        pct_completitud = valores_poblados / total_registros
        valores_faltantes = total_registros - valores_poblados

        reporte_completitud.append(
            {
                "variable": col,
                "valores_poblados": valores_poblados,
                "valores_faltantes": valores_faltantes,
                "pct_completitud": round(pct_completitud, 4),
                "cumple_umbral": pct_completitud >= umbral_completitud,
            }
        )

        if pct_completitud < umbral_completitud:
            variables_excluidas_completitud.append(col)

    df_reporte_completitud = pd.DataFrame(reporte_completitud)

    # Excluir variables que no cumplen completitud
    cols_numericas_validas = [
        col for col in cols_numericas if col not in variables_excluidas_completitud
    ]

    # ========================================================================
    # ETAPA 2: TRATAMIENTO DE VALORES ATÍPICOS (Winsorización P25-P95)
    # ========================================================================

    reporte_atipicos = []

    for col in cols_numericas_validas:
        # Calcular percentiles
        p_inf = df_procesado[col].quantile(percentil_inferior)
        p_sup = df_procesado[col].quantile(percentil_superior)

        # Contar valores fuera de rango
        valores_menores = (df_procesado[col] < p_inf).sum()
        valores_mayores = (df_procesado[col] > p_sup).sum()
        total_ajustados = valores_menores + valores_mayores

        # Aplicar winsorización (reemplazar valores extremos por límites)
        valores_originales_min = df_procesado[col].min()
        valores_originales_max = df_procesado[col].max()

        df_procesado[col] = df_procesado[col].clip(lower=p_inf, upper=p_sup)

        reporte_atipicos.append(
            {
                "variable": col,
                "min_original": valores_originales_min,
                "max_original": valores_originales_max,
                f"p{int(percentil_inferior * 100)}": p_inf,
                f"p{int(percentil_superior * 100)}": p_sup,
                "valores_ajustados_inferior": valores_menores,
                "valores_ajustados_superior": valores_mayores,
                "total_ajustados": total_ajustados,
                "pct_ajustados": round(total_ajustados / len(df_procesado), 4),
            }
        )

    df_reporte_atipicos = pd.DataFrame(reporte_atipicos)

    # ========================================================================
    # ETAPA 3: VALIDACIÓN DE VARIABILIDAD (>= 30%)
    # ========================================================================

    reporte_variabilidad = []
    variables_excluidas_variabilidad = []

    for col in cols_numericas_validas:
        # Calcular percentiles para evaluar distribución
        p0 = df_procesado[col].min()
        p25 = df_procesado[col].quantile(0.25)
        p50 = df_procesado[col].median()
        p75 = df_procesado[col].quantile(0.75)
        p100 = df_procesado[col].max()

        # Calcular coeficiente de variación
        media = df_procesado[col].mean()
        std = df_procesado[col].std()
        cv = std / media if media != 0 else 0

        # Calcular rango intercuartílico relativo (IQR/mediana)
        iqr = p75 - p25
        iqr_relativo = iqr / p50 if p50 != 0 else 0

        # Criterio de variabilidad: CV >= umbral
        cumple_variabilidad = cv >= umbral_variabilidad

        reporte_variabilidad.append(
            {
                "variable": col,
                "min": p0,
                "p25": p25,
                "p50_mediana": p50,
                "p75": p75,
                "max": p100,
                "media": media,
                "std": std,
                "cv": round(cv, 4),
                "iqr_relativo": round(iqr_relativo, 4),
                "cumple_variabilidad": cumple_variabilidad,
            }
        )

        if not cumple_variabilidad:
            variables_excluidas_variabilidad.append(col)

    df_reporte_variabilidad = pd.DataFrame(reporte_variabilidad)

    # Excluir variables con baja variabilidad
    cols_numericas_finales = [
        col for col in cols_numericas_validas if col not in variables_excluidas_variabilidad
    ]

    # ========================================================================
    # RESUMEN FINAL
    # ========================================================================

    # DataFrame final solo con columnas validadas
    columnas_finales = [col_id] + cols_numericas_finales
    df_limpio = df_procesado[columnas_finales].copy()

    # Lista total de variables excluidas
    variables_excluidas_total = list(
        set(variables_excluidas_completitud + variables_excluidas_variabilidad)
    )

    # Retornar resultados
    return {
        "df_limpio": df_limpio,
        "reporte_completitud": df_reporte_completitud,
        "reporte_atipicos": df_reporte_atipicos,
        "reporte_variabilidad": df_reporte_variabilidad,
        "variables_excluidas": variables_excluidas_total,
        "variables_finales": cols_numericas_finales,
        "metricas": {
            "total_inicial": len(cols_numericas),
            "total_final": len(cols_numericas_finales),
            "excluidas_completitud": len(variables_excluidas_completitud),
            "excluidas_variabilidad": len(variables_excluidas_variabilidad),
            "tasa_retencion": len(cols_numericas_finales) / len(cols_numericas)
            if len(cols_numericas) > 0
            else 0,
        },
    }


def crear_variables_combinadas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creacion de nuevas variables combinadas.

    Args:
    df : DataFrame con variables base

    Returns:
    DataFrame con variables base + variables combinadas
    """

    df_features = df.copy()

    # 1. Eficiencia Comercial
    df_features["valor_por_stock"] = df_features["price_sum"] / (df_features["stock_sum"] + 1)
    df_features["rotacion_estimada"] = df_features["price_sum"] / (
        df_features["stock_sum"] * df_features["price_mean"] + 1
    )
    df_features["stock_por_producto"] = df_features["stock_sum"] / (
        df_features["num_productos"] + 1
    )
    df_features["concentracion_premium"] = df_features["price_max"] / (df_features["price_sum"] + 1)

    # 2. Estrategia de Producto
    df_features["productos_por_categoria"] = df_features["num_productos"] / (
        df_features["num_categorias"] + 1
    )
    df_features["indice_especializacion"] = (1 - df_features["entropia_categorias"]) * (
        1 / (df_features["num_categorias"] + 1)
    )
    df_features["amplitud_precios"] = df_features["price_max"] - df_features["price_median"]

    # 3. Posicionamiento
    df_features["ratio_new_used"] = df_features["condition_new"] / (
        df_features["condition_used"] + 0.01
    )
    df_features["reputacion_ponderada"] = df_features["seller_reputation_ordinal"] * np.log1p(
        df_features["num_productos"]
    )

    # 4. Concentración de Categorías
    df_features["concentracion_top_categorias"] = (
        df_features["pct_valor_deportes"]
        + df_features["pct_valor_salud"]
        + df_features["pct_valor_juguetes_y_juegos"]
        + df_features["pct_valor_accesorios_para_autos_y_camion"]
    )
    df_features["diversificacion_otras"] = 1 - df_features["concentracion_top_categorias"]
    df_features["categoria_dominante_valor"] = df_features[
        [
            "pct_valor_deportes",
            "pct_valor_salud",
            "pct_valor_juguetes_y_juegos",
            "pct_valor_accesorios_para_autos_y_camion",
        ]
    ].max(axis=1)

    return df_features


def reducir_redundancia_correlacion(
    df: pd.DataFrame,
    col_id: str = "seller_nickname",
    umbral: float = 0.85,
    variables_protegidas: list[str] | None = None,
) -> dict:
    """
    Reduce redundancia eliminando variables altamente correlacionadas.

    Args:
    -----------
    df : DataFrame a nivel seller con variables numéricas
    col_id : Columna identificadora (excluida del análisis)
    umbral : Umbral de correlación para considerar redundancia (0-1)
    variables_protegidas : Lista de variables que nunca serán eliminadas,
                           incluso si tienen alta correlación con otras

    Returns:
    --------
    dict con:
        - 'df_limpio': DataFrame sin variables redundantes
        - 'variables_eliminadas': Lista de variables eliminadas
        - 'variables_retenidas': Lista de variables retenidas
        - 'matriz_correlacion': Matriz de correlación original
        - 'pares_redundantes': Detalles de pares correlacionados
    """

    df_analisis = df.copy()
    protegidas = set(variables_protegidas or [])

    # Identificar columnas numéricas
    cols_numericas = df_analisis.select_dtypes(include=[np.number]).columns.tolist()
    if col_id in cols_numericas:
        cols_numericas.remove(col_id)

    # Calcular matriz de correlación
    df_numeric = df_analisis[cols_numericas].copy()
    matriz_corr = df_numeric.corr(method="pearson").abs()

    # Identificar y eliminar variables redundantes
    n_vars = len(matriz_corr.columns)
    pares_redundantes = []
    variables_a_eliminar = set()

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            corr_valor = matriz_corr.iloc[i, j]

            if corr_valor > umbral:
                var1 = matriz_corr.columns[i]
                var2 = matriz_corr.columns[j]

                # Si ambas están protegidas, conservar las dos
                if var1 in protegidas and var2 in protegidas:
                    continue

                # Si una está protegida, eliminar siempre la otra
                if var1 in protegidas:
                    var_eliminar, var_retener = var2, var1
                    corr_prom_eliminada = matriz_corr[var2].drop(var2).mean()
                elif var2 in protegidas:
                    var_eliminar, var_retener = var1, var2
                    corr_prom_eliminada = matriz_corr[var1].drop(var1).mean()
                else:
                    # Lógica original: eliminar la más redundante
                    corr_promedio_var1 = matriz_corr[var1].drop(var1).mean()
                    corr_promedio_var2 = matriz_corr[var2].drop(var2).mean()
                    if corr_promedio_var1 > corr_promedio_var2:
                        var_eliminar, var_retener = var1, var2
                        corr_prom_eliminada = corr_promedio_var1
                    else:
                        var_eliminar, var_retener = var2, var1
                        corr_prom_eliminada = corr_promedio_var2

                variables_a_eliminar.add(var_eliminar)

                pares_redundantes.append(
                    {
                        "variable_eliminada": var_eliminar,
                        "variable_retenida": var_retener,
                        "correlacion_par": round(corr_valor, 4),
                        "corr_promedio_eliminada": round(corr_prom_eliminada, 4),
                    }
                )

    # Listas finales
    variables_eliminadas = sorted(list(variables_a_eliminar))
    variables_retenidas = [col for col in cols_numericas if col not in variables_eliminadas]

    # DataFrame limpio
    columnas_finales = [col_id] + variables_retenidas
    df_limpio = df_analisis[columnas_finales].copy()

    return {
        "df_limpio": df_limpio,
        "variables_eliminadas": variables_eliminadas,
        "variables_retenidas": variables_retenidas,
        "matriz_correlacion": matriz_corr,
        "pares_redundantes": pares_redundantes,
    }


CLUSTER_NAMES = {
    0: "Exploradores",
    1: "Exclusivos",
    2: "Consolidados",
    3: "Prometedores",
}


def construir_perfil_seller_agente(
    df_productos: pd.DataFrame,
    df_clusters: pd.DataFrame,
    col_seller: str = "seller_nickname",
    col_precio: str = "price",
    col_stock: str = "stock",
    col_categoria: str = "category_name",
    col_reputacion: str = "seller_reputation",
    col_cluster: str = "cluster",
    top_n_categorias: int = 5,
) -> pd.DataFrame:
    """
    Construye un dataset a nivel seller con los atributos requeridos por el agente generativo.

    Las categorías más representativas de cada seller se determinan ordenando por la
    métrica ``precio × stock`` (valor potencial de inventario) y seleccionando el top N.
    El resultado se presenta como un campo concatenado (``categorias_seller``).

    Args:
        df_productos: DataFrame a nivel producto con columnas de precio, stock,
                      categoría y reputación del seller.
        df_clusters: DataFrame a nivel seller con la asignación de clúster.
                     Debe contener ``col_seller`` y ``col_cluster``.
        col_seller: Nombre de la columna identificadora del seller.
        col_precio: Nombre de la columna de precio unitario.
        col_stock: Nombre de la columna de stock disponible.
        col_categoria: Nombre de la columna de categoría del producto.
        col_reputacion: Nombre de la columna de reputación del seller.
        col_cluster: Nombre de la columna de clúster en ``df_clusters``.
        top_n_categorias: Número de categorías a incluir en el campo concatenado.

    Returns:
        DataFrame a nivel seller con las columnas:
        ``seller_nickname``, ``stock``, ``price``, ``categorias_seller``,
        ``reputacion``, ``cluster``, ``cluster_name``.
    """
    df = df_productos.copy()

    # Métrica de valor potencial por producto
    df["_valor"] = df[col_precio] * df[col_stock]

    # Top N categorías por seller ordenadas por precio × stock
    valor_por_categoria = (
        df.groupby([col_seller, col_categoria])["_valor"].sum().reset_index(name="_valor_cat")
    )

    def _top_categorias(group: pd.DataFrame) -> str:
        top = group.nlargest(top_n_categorias, "_valor_cat")[col_categoria].tolist()
        return ", ".join(top)

    categorias_seller = (
        valor_por_categoria.groupby(col_seller)
        .apply(_top_categorias)
        .reset_index(name="categorias_seller")
    )

    # Agregados numéricos a nivel seller
    agregados = (
        df.groupby(col_seller)
        .agg(
            stock=(col_stock, "sum"),
            price=(col_precio, "mean"),
        )
        .reset_index()
    )

    # Reputación predominante
    reputacion = (
        df.groupby(col_seller)[col_reputacion]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        .reset_index(name="reputacion")
    )

    # Integración
    perfil = agregados.merge(reputacion, on=col_seller).merge(categorias_seller, on=col_seller)

    # Incorporar clúster y nombre del clúster
    df_clusters = df_clusters[[col_seller, col_cluster]].copy()
    perfil = perfil.merge(df_clusters, on=col_seller, how="left")
    perfil["cluster_name"] = perfil[col_cluster].map(CLUSTER_NAMES)

    # Ordenar y renombrar columnas de salida
    perfil = perfil.rename(columns={col_cluster: "cluster"})
    perfil = perfil[
        [
            col_seller,
            "stock",
            "price",
            "categorias_seller",
            "reputacion",
            "cluster",
            "cluster_name",
        ]
    ].reset_index(drop=True)

    logger.info(
        "Perfil de sellers construido: %d sellers, top %d categorías por seller.",
        len(perfil),
        top_n_categorias,
    )
    return perfil


def scale_features(features: pd.DataFrame) -> tuple[pd.DataFrame, RobustScaler]:
    """Escalar la matriz de features construida con RobustScaler.

    Args:
        features: Matriz de features numérica.

    Returns:
        DataFrame escalado y scaler ajustado.
    """
    scaler = RobustScaler()
    scaled_values = scaler.fit_transform(features)
    scaled = pd.DataFrame(scaled_values, index=features.index, columns=features.columns)
    return scaled, scaler
