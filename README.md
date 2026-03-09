# Segmentación de Vendedores – Challenge MercadoLibre

Pipeline de clustering y asesor de estrategia comercial basado en GenAI para la segmentación de vendedores de MercadoLibre.

## Descripción general

Este proyecto segmenta vendedores en función de su catálogo y comportamiento de precios, y luego utiliza un LLM para generar estrategias comerciales personalizadas por segmento. El trabajo sigue la metodología **CRISP-DM** distribuida en seis notebooks.

| Fase | Notebook |
|------|----------|
| Comprensión del negocio | `01_business_understanding.ipynb` |
| Comprensión de los datos (EDA) | `02_data_understanding.ipynb` |
| Preparación de los datos | `03_data_preparation.ipynb` |
| Modelado – Clustering | `04_modeling_clustering.ipynb` |
| Evaluación | `05_evaluation.ipynb` |
| Extensión GenAI | `06_genai_extension.ipynb` |

## Requisitos

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (gestor de paquetes)

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/AnaUrcuqui/seller-segmentation-meli.git
cd seller-segmentation

# 2. Instalar dependencias
make install

# 3. Configurar variables de entorno
cp .env.example .env
# Editar .env y añadir la clave ANTHROPIC_API_KEY
```

## Dataset

El dataset **no está incluido en el repositorio**. Colocar el archivo CSV en la siguiente ruta antes de ejecutar los notebooks:

```
data/raw/df_challenge_meli - df_challenge_meli.csv
```

Una vez ubicado el archivo, los notebooks lo leen automáticamente sin ningún paso adicional.

## Reproducir el análisis

> **Importante:** Ejecutar todos los comandos desde la **raíz del repositorio** (`seller-segmentation/`). Los notebooks detectan automáticamente el directorio correcto, pero si los abres desde otra ubicación sin usar `make notebooks`, podrías ver errores de rutas.

```bash
# Abrir JupyterLab y ejecutar los notebooks del 01 al 06 en orden
make notebooks
```

Para ejecutar cada notebook de forma individual desde la terminal:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_data_understanding.ipynb
```

## Estructura del proyecto

```
seller-segmentation/
├── .github/workflows/     # CI (lint, tests, validación de notebooks) + CD (reportes HTML)
├── data/
│   ├── raw/               # CSV original (no incluido en el repositorio)
│   ├── interim/           # Artefactos intermedios
│   └── processed/         # Matrices de features, etiquetas, estrategias
├── notebooks/             # Notebooks CRISP-DM (01–06)
├── src/seller_segmentation/
│   ├── data/              # Carga y preprocesamiento de datos
│   ├── features/          # Ingeniería de features
│   ├── models/            # Clustering + asesor GenAI
│   └── viz/               # Utilidades de visualización
├── tests/                 # Tests unitarios (pytest)
├── scripts/               # Scripts auxiliares (validación de notebooks)
├── reports/figures/       # Gráficos generados automáticamente
├── Makefile
└── pyproject.toml
```

## Ejecutar los tests

```bash
make test
```

## Extensión GenAI

Este proyecto implementa la **Opción B – Asesor de Estrategia Comercial Generativa**:

Dado el cluster asignado a un vendedor y su perfil de métricas, un LLM (`claude-sonnet-4-6`) genera una estrategia comercial personalizada que incluye recomendaciones de campañas, mejoras de catálogo y KPIs de seguimiento.

Ver `notebooks/06_genai_extension.ipynb` y `src/seller_segmentation/models/genai_extension.py`.

## Versión

**v0.1.0** — Pipeline completo CRISP-DM con las siguientes capacidades:

- Preprocesamiento y normalización de datos de vendedores MercadoLibre
- Ingeniería de features sobre catálogo y comportamiento de precios
- Clustering con K-Means (selección automática de k via Elbow + Silhouette)
- Generación de estrategias comerciales por segmento vía `claude-sonnet-4-6`

## CI/CD

- **CI** (`ci.yml`): se ejecuta en cada push/PR — linting (ruff), verificación de tipos (mypy), tests unitarios, validación de notebooks.
- **CD** (`cd.yml`): convierte los notebooks a HTML y los publica en GitHub Pages al hacer merge a `main`.
