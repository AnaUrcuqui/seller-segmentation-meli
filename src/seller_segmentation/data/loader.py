"""Utilidades para la adquisición de datos."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_RAW_PATH = (
    Path(__file__).parents[3] / "data" / "raw" / "df_challenge_meli - df_challenge_meli.csv"
)


def load_df(path: Path = _DEFAULT_RAW_PATH) -> pd.DataFrame:
    """Carga el CSV del challenger en un DataFrame.

    Args:
        path: Ruta al archivo CSV.

    Returns:
        DataFrame con los datos del challenger sin procesar.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado en {path}. Asegúrese de que el archivo esté en data/raw/."
        )
    df = pd.read_csv(path, sep=",", header=0, low_memory=False)
    logger.info("Cargadas %d filas y %d columnas desde %s.", *df.shape, path)
    return df
