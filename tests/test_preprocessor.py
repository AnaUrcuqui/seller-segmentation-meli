"""Tests para las utilidades de preprocesamiento de datos."""

import pandas as pd
import pytest

from seller_segmentation.data.preprocessor import (
    clean_challenger_df,
    identificar_columnas_mono_categoria,
)


@pytest.fixture()
def raw_items() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "seller_nickname": [
                "vendedor_a",
                "vendedor_a",
                "vendedor_b",
                "vendedor_b",
                "vendedor_b",
                None,
            ],
            "price": [100.0, 200.0, 50.0, 75.0, 125.0, 300.0],
            "stock": [10, 5, 20, 15, 8, 3],
            "category_name": ["electronics", "books", "clothing", "clothing", "books", "toys"],
            "condition": ["new", "used", "new", "new", "used", "new"],
            "seller_reputation": ["green", "green", "yellow", "yellow", "yellow", "red"],
            "logistic_type": ["xd", "xd", "ds", "ds", "ds", "xd"],
        }
    )


def test_clean_drops_null_seller(raw_items: pd.DataFrame) -> None:
    cleaned = clean_challenger_df(raw_items)
    assert cleaned["seller_nickname"].notna().all()
    assert len(cleaned) < len(raw_items)


def test_clean_drops_duplicates() -> None:
    df = pd.DataFrame(
        {
            "seller_nickname": ["vendedor_a", "vendedor_a"],
            "price": [100.0, 100.0],
            "stock": [10, 10],
            "category_name": ["electronics", "electronics"],
            "condition": ["new", "new"],
            "seller_reputation": ["green", "green"],
            "logistic_type": ["xd", "xd"],
        }
    )
    cleaned = clean_challenger_df(df)
    assert len(cleaned) == 1


def test_clean_returns_dataframe(raw_items: pd.DataFrame) -> None:
    cleaned = clean_challenger_df(raw_items)
    assert isinstance(cleaned, pd.DataFrame)


def test_identificar_mono_categoria_columns(raw_items: pd.DataFrame) -> None:
    cleaned = clean_challenger_df(raw_items)
    resumen, detalle = identificar_columnas_mono_categoria(
        cleaned, "seller_nickname", ["category_name", "condition"]
    )
    assert "columna" in resumen.columns
    assert set(resumen["columna"]) == {"category_name", "condition"}


def test_identificar_mono_categoria_detalle_keys(raw_items: pd.DataFrame) -> None:
    cleaned = clean_challenger_df(raw_items)
    _, detalle = identificar_columnas_mono_categoria(
        cleaned, "seller_nickname", ["category_name", "condition"]
    )
    assert "category_name" in detalle
    assert "condition" in detalle
