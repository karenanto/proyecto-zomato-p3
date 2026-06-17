"""
load.py
=======
Módulo de carga del pipeline ETL Zomato.

Responsabilidades:
    - Guardar el DataFrame procesado como CSV
    - Serializar modelos entrenados con joblib
    - Exportar métricas como JSON
    - Registrar en log cada archivo generado
"""

import json
import logging
import os
import joblib
import pandas as pd

logger = logging.getLogger(__name__)

PROCESSED_PATH = os.getenv("DATA_PROCESSED_PATH", "data/processed/")


def _asegurar_directorio(path: str) -> None:
    """Crea el directorio si no existe."""
    os.makedirs(path, exist_ok=True)


def guardar_dataframe(df: pd.DataFrame, filename: str = "zomato_final.csv") -> str:
    """
    Guarda el DataFrame procesado como CSV.

    Args:
        df:       DataFrame a exportar.
        filename: Nombre del archivo de salida.

    Returns:
        Ruta completa del archivo guardado.
    """
    _asegurar_directorio(PROCESSED_PATH)
    path = os.path.join(PROCESSED_PATH, filename)
    df.to_csv(path, index=False)
    logger.info("Load — DataFrame guardado: %s (%d filas)", path, len(df))
    return path


def guardar_labels_clusters(
    df: pd.DataFrame,
    labels,
    filename: str = "cluster_labels.csv",
) -> str:
    """
    Guarda las etiquetas de cluster junto a un identificador de restaurante.

    Args:
        df:       DataFrame original (para extraer 'name' si existe).
        labels:   Array de etiquetas de cluster.
        filename: Nombre del archivo de salida.

    Returns:
        Ruta completa del archivo guardado.
    """
    _asegurar_directorio(PROCESSED_PATH)
    path = os.path.join(PROCESSED_PATH, filename)

    df_labels = pd.DataFrame({"cluster": labels})
    if "name" in df.columns:
        df_labels.insert(0, "name", df["name"].values[: len(labels)])

    df_labels.to_csv(path, index=False)
    logger.info("Load — Etiquetas de cluster guardadas: %s", path)
    return path


def guardar_metricas(metricas_clustering: dict, metricas_sup: dict) -> str:
    """
    Exporta todas las métricas del pipeline a un archivo JSON.

    Args:
        metricas_clustering: Dict retornado por aplicar_kmeans().
        metricas_sup:        Dict retornado por entrenar_modelos_supervisados().

    Returns:
        Ruta completa del archivo JSON guardado.
    """
    _asegurar_directorio(PROCESSED_PATH)
    path = os.path.join(PROCESSED_PATH, "model_metrics.json")

    metricas_totales = {
        "clustering": metricas_clustering,
        "regresion":  metricas_sup,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metricas_totales, f, indent=2, ensure_ascii=False)

    logger.info("Load — Métricas guardadas: %s", path)
    return path


def guardar_modelos(
    pipeline_pca,
    modelo_kmeans,
    modelo_regresion,
) -> dict:
    """
    Serializa los modelos entrenados con joblib para reutilización en la API.

    Args:
        pipeline_pca:      Pipeline sklearn (StandardScaler + PCA).
        modelo_kmeans:     Modelo KMeans entrenado.
        modelo_regresion:  Mejor modelo de regresión (Ridge o RF).

    Returns:
        Diccionario con las rutas de cada archivo guardado.
    """
    _asegurar_directorio(PROCESSED_PATH)

    rutas = {
        "pipeline_pca":     os.path.join(PROCESSED_PATH, "pipeline_pca.pkl"),
        "modelo_kmeans":    os.path.join(PROCESSED_PATH, "modelo_kmeans.pkl"),
        "modelo_regresion": os.path.join(PROCESSED_PATH, "modelo_regresion.pkl"),
    }

    joblib.dump(pipeline_pca,     rutas["pipeline_pca"])
    joblib.dump(modelo_kmeans,    rutas["modelo_kmeans"])
    joblib.dump(modelo_regresion, rutas["modelo_regresion"])

    for nombre, ruta in rutas.items():
        logger.info("Load — Modelo '%s' guardado: %s", nombre, ruta)

    return rutas
