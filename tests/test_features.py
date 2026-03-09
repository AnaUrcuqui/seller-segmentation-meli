"""Tests para la ingeniería de features."""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler

from seller_segmentation.features.builder import (
    agregar_variables_numericas,
    reducir_redundancia_correlacion,
    scale_features,
    validar_y_limpiar_variables_numericas,
)


@pytest.fixture()
def items_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "seller_nickname": ["A", "A", "B", "B", "B"],
            "price": [100.0, 200.0, 50.0, 75.0, 125.0],
            "stock": [10, 5, 20, 15, 8],
        }
    )


@pytest.fixture()
def seller_profile() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "total_items": [10, 5, 50],
            "avg_price": [100.0, 500.0, 30.0],
            "median_price": [90.0, 480.0, 25.0],
            "price_std": [20.0, 100.0, 5.0],
        },
        index=pd.Index(["A", "B", "C"], name="seller_id"),
    )


@pytest.fixture()
def numeric_seller_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 30
    return pd.DataFrame(
        {
            "seller_nickname": [f"seller_{i}" for i in range(n)],
            "price_mean": rng.uniform(10, 1000, n),
            "price_sum": rng.uniform(100, 10000, n),
            "stock_sum": rng.uniform(1, 500, n),
            "num_productos": rng.integers(1, 200, n).astype(float),
        }
    )


# --- agregar_variables_numericas ---


def test_agregar_variables_numericas_shape(items_df: pd.DataFrame) -> None:
    result = agregar_variables_numericas(items_df, "seller_nickname", ["price", "stock"])
    assert "seller_nickname" in result.columns
    assert len(result) == 2


def test_agregar_variables_numericas_conteo(items_df: pd.DataFrame) -> None:
    result = agregar_variables_numericas(
        items_df, "seller_nickname", ["price"], nombre_conteo="num_items"
    )
    row_a = result[result["seller_nickname"] == "A"].iloc[0]
    assert row_a["num_items"] == 2


def test_agregar_variables_numericas_columnas_generadas(items_df: pd.DataFrame) -> None:
    result = agregar_variables_numericas(items_df, "seller_nickname", ["price"])
    generated = [c for c in result.columns if c.startswith("price_")]
    assert len(generated) > 0


# --- scale_features ---


def test_scale_features_shape(seller_profile: pd.DataFrame) -> None:
    scaled, _ = scale_features(seller_profile)
    assert scaled.shape == seller_profile.shape
    assert scaled.index.equals(seller_profile.index)


def test_scale_features_returns_scaler(seller_profile: pd.DataFrame) -> None:
    _, scaler = scale_features(seller_profile)
    assert isinstance(scaler, RobustScaler)


# --- validar_y_limpiar_variables_numericas ---


def test_validar_retorna_claves_esperadas(numeric_seller_df: pd.DataFrame) -> None:
    result = validar_y_limpiar_variables_numericas(numeric_seller_df, col_id="seller_nickname")
    for key in ("df_limpio", "variables_finales", "variables_excluidas", "metricas"):
        assert key in result


def test_validar_df_limpio_contiene_id(numeric_seller_df: pd.DataFrame) -> None:
    result = validar_y_limpiar_variables_numericas(numeric_seller_df, col_id="seller_nickname")
    assert "seller_nickname" in result["df_limpio"].columns


def test_validar_metricas_tasa_retencion(numeric_seller_df: pd.DataFrame) -> None:
    result = validar_y_limpiar_variables_numericas(numeric_seller_df, col_id="seller_nickname")
    tasa = result["metricas"]["tasa_retencion"]
    assert 0.0 <= tasa <= 1.0


def test_validar_excluye_baja_variabilidad() -> None:
    df = pd.DataFrame(
        {
            "seller_nickname": [f"s{i}" for i in range(20)],
            "constante": [1.0] * 20,
            "variable": list(range(20)),
        }
    )
    result = validar_y_limpiar_variables_numericas(
        df, col_id="seller_nickname", umbral_variabilidad=0.1
    )
    assert "constante" in result["variables_excluidas"]


# --- reducir_redundancia_correlacion ---


def test_reducir_retorna_claves_esperadas(numeric_seller_df: pd.DataFrame) -> None:
    result = reducir_redundancia_correlacion(numeric_seller_df, col_id="seller_nickname")
    for key in ("df_limpio", "variables_eliminadas", "variables_retenidas"):
        assert key in result


def test_reducir_elimina_columna_correlada() -> None:
    df = pd.DataFrame(
        {
            "seller_nickname": [f"s{i}" for i in range(10)],
            "x": [float(i) for i in range(10)],
            "y": [float(i) * 2 for i in range(10)],  # perfectamente correlada con x
            "z": [float(i) ** 0.5 for i in range(10)],
        }
    )
    result = reducir_redundancia_correlacion(df, col_id="seller_nickname", umbral=0.95)
    assert len(result["variables_eliminadas"]) >= 1


def test_reducir_df_limpio_mantiene_id(numeric_seller_df: pd.DataFrame) -> None:
    result = reducir_redundancia_correlacion(numeric_seller_df, col_id="seller_nickname")
    assert "seller_nickname" in result["df_limpio"].columns
