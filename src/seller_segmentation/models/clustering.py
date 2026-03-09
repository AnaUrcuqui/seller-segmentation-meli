"""Pipeline de clustering de vendedores."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

_RANDOM_STATE = 42


class SellerClusterer:
    """Envuelve KMeans con selección de k por el método del codo y diagnósticos de clusters.

    Uso::
        clusterer = SellerClusterer(k_range=(2, 12))
        clusterer.fit(scaled_features)
        labels = clusterer.labels_
        print(clusterer.metrics_)
    """

    def __init__(self, k_range: tuple[int, int] = (2, 12)) -> None:
        self.k_range = k_range
        self.best_k_: int | None = None
        self.model_: KMeans | None = None
        self.labels_: pd.Series | None = None
        self.inertias_: dict[int, float] = {}
        self.metrics_: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Ajuste
    # ------------------------------------------------------------------

    def fit(self, features: pd.DataFrame) -> SellerClusterer:
        """Ajusta KMeans tras seleccionar el mejor k mediante el método del codo.

        Args:
            features: Matriz de features escalada con índice de vendedores.

        Returns:
            La propia instancia.
        """
        self.inertias_ = self._sweep_k(features)
        self.best_k_ = self._elbow_k(self.inertias_)
        logger.info("k=%d clusters seleccionados.", self.best_k_)

        self.model_ = KMeans(n_clusters=self.best_k_, random_state=_RANDOM_STATE, n_init="auto")
        raw_labels = self.model_.fit_predict(features)
        self.labels_ = pd.Series(raw_labels, index=features.index, name="cluster")

        self.metrics_ = {
            "silhouette": silhouette_score(features, raw_labels),
            "calinski_harabasz": calinski_harabasz_score(features, raw_labels),
            "davies_bouldin": davies_bouldin_score(features, raw_labels),
        }
        logger.info("Métricas de clustering: %s", self.metrics_)
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Asigna nuevos vendedores al centroide más cercano.

        Args:
            features: Matriz de features escalada para nuevos vendedores.

        Returns:
            Serie con las etiquetas de cluster.
        """
        if self.model_ is None:
            raise RuntimeError("Llamar a fit() antes de predict().")
        raw = self.model_.predict(features)
        return pd.Series(raw, index=features.index, name="cluster")

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        joblib.dump(self, path)
        logger.info("Modelo guardado en %s.", path)

    @classmethod
    def load(cls, path: Path | str) -> SellerClusterer:
        instance: SellerClusterer = joblib.load(path)
        return instance

    # ------------------------------------------------------------------
    # Métodos internos
    # ------------------------------------------------------------------

    def _sweep_k(self, features: pd.DataFrame) -> dict[int, float]:
        inertias = {}
        k_min, k_max = self.k_range
        for k in range(k_min, k_max + 1):
            km = KMeans(n_clusters=k, random_state=_RANDOM_STATE, n_init="auto")
            km.fit(features)
            inertias[k] = float(km.inertia_)
            logger.debug("k=%d → inercia=%.2f", k, km.inertia_)
        return inertias

    @staticmethod
    def _elbow_k(inertias: dict[int, float]) -> int:
        """Identifica el punto del codo usando la segunda derivada máxima."""
        ks = sorted(inertias)
        vals = np.array([inertias[k] for k in ks])
        if len(vals) < 3:
            return ks[0]
        second_deriv = np.diff(vals, n=2)
        elbow_idx = int(np.argmax(second_deriv)) + 1  # offset por pérdida en diff
        return ks[elbow_idx]


class SellerDBSCAN:
    """Envuelve DBSCAN con selección automática de eps y diagnósticos de clusters.

    Usa la curva k-distancias para estimar eps óptimo automáticamente.

    Uso::
        dbscan = SellerDBSCAN(min_samples=5)
        dbscan.fit(scaled_features)
        labels = dbscan.labels_
        print(dbscan.metrics_)
        print(f"Clusters encontrados: {dbscan.n_clusters_}, Ruido: {dbscan.noise_ratio_:.1%}")
    """

    def __init__(self, eps: float | None = None, min_samples: int = 5) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.eps_: float | None = None
        self.model_: DBSCAN | None = None
        self.labels_: pd.Series | None = None
        self.n_clusters_: int | None = None
        self.noise_ratio_: float | None = None
        self.metrics_: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Ajuste
    # ------------------------------------------------------------------

    def fit(self, features: pd.DataFrame) -> SellerDBSCAN:
        """Ajusta DBSCAN, estimando eps automáticamente si no se provee.

        Args:
            features: Matriz de features escalada con índice de vendedores.

        Returns:
            La propia instancia.
        """
        self.eps_ = self.eps if self.eps is not None else self._estimate_eps(features)
        logger.info("eps=%.4f, min_samples=%d", self.eps_, self.min_samples)

        self.model_ = DBSCAN(eps=self.eps_, min_samples=self.min_samples)
        raw_labels = self.model_.fit_predict(features)
        self.labels_ = pd.Series(raw_labels, index=features.index, name="cluster")

        mask_core = raw_labels != -1
        self.n_clusters_ = int(len(set(raw_labels[mask_core])))
        self.noise_ratio_ = float((raw_labels == -1).mean())
        logger.info(
            "Clusters encontrados: %d, puntos de ruido: %.1f%%",
            self.n_clusters_,
            self.noise_ratio_ * 100,
        )

        if self.n_clusters_ >= 2 and mask_core.sum() > 0:
            features_core = features.values[mask_core]
            labels_core = raw_labels[mask_core]
            self.metrics_ = {
                "silhouette": silhouette_score(features_core, labels_core),
                "calinski_harabasz": calinski_harabasz_score(features_core, labels_core),
                "davies_bouldin": davies_bouldin_score(features_core, labels_core),
            }
            logger.info("Métricas de clustering: %s", self.metrics_)
        else:
            logger.warning("Menos de 2 clusters; métricas no calculadas.")
            self.metrics_ = {}

        return self

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        joblib.dump(self, path)
        logger.info("Modelo guardado en %s.", path)

    @classmethod
    def load(cls, path: Path | str) -> SellerDBSCAN:
        instance: SellerDBSCAN = joblib.load(path)
        return instance

    # ------------------------------------------------------------------
    # Métodos internos
    # ------------------------------------------------------------------

    def _estimate_eps(self, features: pd.DataFrame) -> float:
        """Estima eps usando el método k-distancias (codo en distancias ordenadas).

        Calcula la distancia al k-ésimo vecino más cercano para cada punto,
        las ordena y busca el codo con la segunda derivada máxima.
        """
        k = self.min_samples
        nbrs = NearestNeighbors(n_neighbors=k).fit(features)
        distances, _ = nbrs.kneighbors(features)
        k_distances = np.sort(distances[:, -1])[::-1]

        second_deriv = np.diff(k_distances, n=2)
        elbow_idx = int(np.argmax(second_deriv))
        eps = float(k_distances[elbow_idx])
        logger.info("eps estimado por k-distancias: %.4f", eps)
        return eps
