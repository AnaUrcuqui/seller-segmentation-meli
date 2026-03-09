"""Tests para SellerClusterer."""

import numpy as np
import pandas as pd
import pytest

from seller_segmentation.models.clustering import SellerClusterer

# El modelo real opera con ~20 features escaladas (stock, precio, logística,
# reputación, categorías, variables combinadas) y k=4 (Silhouette≈0.58 en prod).
# El fixture replica esa dimensionalidad con 4 clusters bien separados para que
# el método del codo identifique k=4 de forma confiable.
N_FEATURES = 20
N_PER_CLUSTER = 40
CENTERS = [0.0, 10.0, 20.0, 30.0]


@pytest.fixture()
def scaled_features() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    groups = [rng.normal(loc=c, scale=0.5, size=(N_PER_CLUSTER, N_FEATURES)) for c in CENTERS]
    data = np.vstack(groups)
    cols = [f"f{i}" for i in range(N_FEATURES)]
    index = pd.Index(
        [f"seller_{i}" for i in range(N_PER_CLUSTER * len(CENTERS))],
        name="seller_id",
    )
    return pd.DataFrame(data, index=index, columns=cols)


def test_fit_returns_self(scaled_features: pd.DataFrame) -> None:
    clusterer = SellerClusterer(k_range=(2, 8))
    result = clusterer.fit(scaled_features)
    assert result is clusterer


def test_labels_aligned_with_index(scaled_features: pd.DataFrame) -> None:
    clusterer = SellerClusterer(k_range=(2, 8))
    clusterer.fit(scaled_features)
    assert clusterer.labels_ is not None
    assert clusterer.labels_.index.equals(scaled_features.index)


def test_silhouette_reasonable(scaled_features: pd.DataFrame) -> None:
    # Con 4 clusters bien separados el elbow elige k=4 y silhouette debe ser alto
    clusterer = SellerClusterer(k_range=(2, 8))
    clusterer.fit(scaled_features)
    assert clusterer.metrics_["silhouette"] > 0.5


def test_predict_shape(scaled_features: pd.DataFrame) -> None:
    clusterer = SellerClusterer(k_range=(2, 8))
    clusterer.fit(scaled_features)
    preds = clusterer.predict(scaled_features.head(10))
    assert len(preds) == 10
