"""Utilidades de visualización reutilizables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

_PALETTE = "Set2"
_FIG_DIR = Path("reports/figures")


def _save(fig: plt.Figure, name: str) -> None:
    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(_FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")


def plot_elbow(inertias: dict[int, float], best_k: int | None = None) -> plt.Figure:
    """Curva del codo: inercia de KMeans frente al número de clusters."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ks, vals = zip(*sorted(inertias.items()), strict=False)
    ax.plot(ks, vals, marker="o", linewidth=2)
    if best_k is not None:
        ax.axvline(best_k, color="crimson", linestyle="--", label=f"k seleccionado={best_k}")
        ax.legend()
    ax.set_xlabel("Número de clusters (k)")
    ax.set_ylabel("Inercia")
    ax.set_title("Curva del codo – KMeans")
    sns.despine()
    _save(fig, "elbow_curve")
    return fig


def plot_cluster_distribution(labels: pd.Series) -> plt.Figure:
    """Gráfico de barras con el número de vendedores por cluster."""
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = labels.value_counts().sort_index()
    counts.plot(kind="bar", ax=ax, color=sns.color_palette(_PALETTE, len(counts)))
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Número de vendedores")
    ax.set_title("Vendedores por cluster")
    ax.tick_params(axis="x", rotation=0)
    sns.despine()
    _save(fig, "cluster_distribution")
    return fig


def plot_feature_importance(
    features: pd.DataFrame,
    labels: pd.Series,
    top_n: int = 10,
) -> plt.Figure:
    """Mapa de calor con los valores medios de features por cluster."""
    df = features.copy()
    df["cluster"] = labels
    means = df.groupby("cluster").mean()

    # Conservar las features más discriminativas (mayor varianza entre clusters)
    variance = means.var(axis=0).nlargest(top_n).index
    means = means[variance]

    fig, ax = plt.subplots(figsize=(12, max(4, len(means) * 0.6)))
    sns.heatmap(
        means.T,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(f"Top {top_n} features – valor medio por cluster")
    ax.set_xlabel("Cluster")
    _save(fig, "feature_heatmap")
    return fig


def plot_kdistances(
    features: pd.DataFrame | np.ndarray,
    min_samples: int = 5,
    eps_estimado: float | None = None,
) -> plt.Figure:
    """Curva k-distancias para estimar eps óptimo de DBSCAN.

    Args:
        features: Matriz de features escalada.
        min_samples: Igual que el parámetro ``min_samples`` de DBSCAN.
        eps_estimado: Si se provee, dibuja una línea horizontal en ese valor.

    Returns:
        Figura de matplotlib.
    """
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(features)
    distances, _ = nbrs.kneighbors(features)
    k_distances = np.sort(distances[:, -1])[::-1]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(k_distances, linewidth=1.5, color="steelblue")
    ax.set_xlabel("Puntos ordenados (desc.)")
    ax.set_ylabel(f"Distancia al {min_samples}º vecino")
    ax.set_title(f"Curva k-distancias – DBSCAN (min_samples={min_samples})")

    if eps_estimado is not None:
        ax.axhline(
            eps_estimado,
            color="crimson",
            linestyle="--",
            label=f"eps estimado = {eps_estimado:.4f}",
        )
        ax.legend()

    sns.despine()
    _save(fig, "kdistances_curve")
    return fig


def visualizar_clustering(df: pd.DataFrame, figsize: tuple[int, int] = (5, 3)) -> plt.Figure:
    """
    Visualiza resultados del clustering.
    """

    df_work = df.copy()
    cols_num = [
        c for c in df_work.columns if c not in ["seller_nickname", "price_median", "cluster"]
    ]
    x_vals = df[cols_num].values
    clusters = df["cluster"].values

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # 1. PCA 2D
    pca = PCA(n_components=2, random_state=42)
    x_pca = pca.fit_transform(x_vals)

    scatter = ax.scatter(
        x_pca[:, 0],
        x_pca[:, 1],
        c=clusters,
        cmap="tab10",
        s=50,
        alpha=0.6,
        edgecolors="k",
        linewidth=0.5,
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("Clusters en Espacio PCA")
    plt.colorbar(scatter, ax=ax, label="Cluster")

    plt.tight_layout()
    return fig


def visualizar_correlaciones(
    resultado: dict, top_n: int = 30, figsize: tuple[int, int] = (16, 6)
) -> plt.Figure:
    """
    Visualiza matriz de correlación y distribución de correlaciones.

    Args:
    -----------
    resultado : Salida de reducir_redundancia_correlacion()
    top_n : Número máximo de variables a mostrar en heatmap
    figsize : Tamaño de la figura
    """

    matriz_corr = resultado["matriz_correlacion"]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 1. Heatmap de correlación
    if len(matriz_corr) > top_n:
        # Seleccionar variables con mayor varianza en correlaciones
        varianza_corr = matriz_corr.var().sort_values(ascending=False)
        top_vars = varianza_corr.head(top_n).index
        matriz_plot = matriz_corr.loc[top_vars, top_vars]
    else:
        matriz_plot = matriz_corr

    sns.heatmap(
        matriz_plot,
        annot=False,
        cmap="RdYlBu_r",
        center=0,
        vmin=0,
        vmax=1,
        square=True,
        ax=axes[0],
        cbar_kws={"label": "Correlación (valor absoluto)"},
    )
    axes[0].set_title(
        f"Matriz de Correlación\n({len(matriz_plot)} variables)", fontsize=13, fontweight="bold"
    )
    axes[0].tick_params(axis="both", labelsize=8)

    # 2. Distribución de correlaciones
    # Extraer triángulo superior (sin diagonal)
    triu_indices = np.triu_indices_from(matriz_corr, k=1)
    correlaciones = matriz_corr.values[triu_indices]

    axes[1].hist(
        correlaciones, bins=50, color="steelblue", alpha=0.7, edgecolor="black", linewidth=0.5
    )
    axes[1].axvline(
        x=0.85, color="red", linestyle="--", linewidth=2.5, label="Umbral 0.85", zorder=5
    )
    axes[1].set_xlabel("Correlación (valor absoluto)", fontsize=12)
    axes[1].set_ylabel("Frecuencia", fontsize=12)
    axes[1].set_title(
        "Distribución de Correlaciones entre Variables", fontsize=13, fontweight="bold"
    )
    axes[1].legend(fontsize=11)
    axes[1].grid(axis="y", alpha=0.3)

    # Añadir estadísticas
    num_total_pares = len(correlaciones)
    num_alta_corr = (correlaciones > 0.85).sum()
    pct_alta_corr = (num_alta_corr / num_total_pares * 100) if num_total_pares > 0 else 0

    texto_stats = f"Pares con corr > 0.85: {num_alta_corr} ({pct_alta_corr:.1f}%)"
    axes[1].text(
        0.98,
        0.95,
        texto_stats,
        transform=axes[1].transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=10,
    )

    plt.tight_layout()
    return fig


def visualizar_reportes_validacion(resultados: dict) -> plt.Figure:
    """
    Genera visualizaciones de los reportes de validación.
    """

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Completitud de variables
    df_comp = resultados["reporte_completitud"].sort_values("pct_completitud")
    colors = ["red" if not x else "green" for x in df_comp["cumple_umbral"]]

    axes[0, 0].barh(range(len(df_comp)), df_comp["pct_completitud"], color=colors, alpha=0.7)
    axes[0, 0].axvline(x=0.90, color="black", linestyle="--", linewidth=2, label="Umbral 90%")
    axes[0, 0].set_yticks(range(len(df_comp)))
    axes[0, 0].set_yticklabels(df_comp["variable"], fontsize=8)
    axes[0, 0].set_xlabel("% Completitud", fontsize=12)
    axes[0, 0].set_title("Completitud de Variables", fontsize=14, fontweight="bold")
    axes[0, 0].legend()
    axes[0, 0].invert_yaxis()

    # 2. Valores atípicos ajustados
    df_atip = (
        resultados["reporte_atipicos"].sort_values("total_ajustados", ascending=False).head(15)
    )
    axes[0, 1].barh(range(len(df_atip)), df_atip["total_ajustados"], color="coral", alpha=0.7)
    axes[0, 1].set_yticks(range(len(df_atip)))
    axes[0, 1].set_yticklabels(df_atip["variable"], fontsize=8)
    axes[0, 1].set_xlabel("Valores Ajustados", fontsize=12)
    axes[0, 1].set_title("Top 15 Variables con Ajuste de Atípicos", fontsize=14, fontweight="bold")
    axes[0, 1].invert_yaxis()

    # 3. Coeficiente de variación
    df_var = resultados["reporte_variabilidad"].sort_values("cv")
    colors_var = ["red" if not x else "green" for x in df_var["cumple_variabilidad"]]

    axes[1, 0].barh(range(len(df_var)), df_var["cv"], color=colors_var, alpha=0.7)
    axes[1, 0].axvline(x=0.30, color="black", linestyle="--", linewidth=2, label="Umbral 30%")
    axes[1, 0].set_yticks(range(len(df_var)))
    axes[1, 0].set_yticklabels(df_var["variable"], fontsize=8)
    axes[1, 0].set_xlabel("Coeficiente de Variación", fontsize=12)
    axes[1, 0].set_title("Variabilidad de Variables (CV)", fontsize=14, fontweight="bold")
    axes[1, 0].legend()
    axes[1, 0].invert_yaxis()

    # 4. Resumen de exclusiones
    metricas = resultados["metricas"]
    labels = ["Variables\nFinales", "Excluidas por\nCompletitud", "Excluidas por\nVariabilidad"]
    valores = [
        metricas["total_final"],
        metricas["excluidas_completitud"],
        metricas["excluidas_variabilidad"],
    ]
    colors_resumen = ["green", "orange", "red"]

    axes[1, 1].bar(labels, valores, color=colors_resumen, alpha=0.7)
    axes[1, 1].set_ylabel("Número de Variables", fontsize=12)
    axes[1, 1].set_title("Resumen de Validación", fontsize=14, fontweight="bold")

    for i, v in enumerate(valores):
        axes[1, 1].text(i, v + 0.5, str(v), ha="center", fontweight="bold")

    plt.tight_layout()
    return fig
