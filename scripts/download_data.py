"""Script auxiliar – verifica que el dataset esté disponible en data/raw/."""

from seller_segmentation.data.loader import load_df

if __name__ == "__main__":
    df = load_df()
    print(f"Dataset disponible: {df.shape[0]:,} filas × {df.shape[1]} columnas")
