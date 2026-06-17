"""
pipeline.py
===========
Orquestador principal del pipeline ETL Zomato.

Ejecuta en orden:
    1. Extract  — 3 fuentes de datos
    2. Transform — limpieza, validación, PCA, clustering, regresión
    3. Load     — guarda resultados, modelos y métricas

Uso:
    python etl/pipeline.py

Variables de entorno (configurables en .env):
    DATA_RAW_PATH       — ruta a los datos crudos (default: data/raw/)
    DATA_PROCESSED_PATH — ruta de salida (default: data/processed/)
    COUNTRIES_API_URL   — endpoint Rest Countries API
    EXCHANGE_API_URL    — endpoint Open Exchange Rates API
"""

import logging
import sys
import time
from dotenv import load_dotenv

load_dotenv()

# ── Logging profesional ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/processed/etl.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("pipeline")

# ── Importaciones internas ────────────────────────────────────────────────────
from extract import (
    extract_zomato_csv,
    extract_country_info,
    extract_exchange_rate,
)
from transform import (
    limpiar_y_enriquecer,
    validar_esquema,
    preparar_features_clustering,
    aplicar_pca,
    aplicar_kmeans,
    entrenar_modelos_supervisados,
)
from load import (
    guardar_dataframe,
    guardar_labels_clusters,
    guardar_metricas,
    guardar_modelos,
)


def run_pipeline() -> dict:
    """
    Ejecuta el pipeline ETL completo de extremo a extremo.

    Returns:
        Diccionario con rutas de archivos generados y métricas obtenidas.

    Raises:
        Exception: Propaga cualquier error crítico tras registrarlo en el log.
    """
    inicio = time.time()
    logger.info("=" * 60)
    logger.info("INICIO DEL PIPELINE ETL — Zomato Analytics")
    logger.info("=" * 60)

    resultados = {}

    # ── EXTRACT ───────────────────────────────────────────────────────────────
    logger.info("── FASE EXTRACT ──")

    try:
        df_raw = extract_zomato_csv()
    except (FileNotFoundError, ValueError) as e:
        logger.critical("Fuente 1 (CSV) falló: %s. Pipeline abortado.", e)
        sys.exit(1)

    try:
        country_info = extract_country_info()
    except Exception as e:
        logger.warning("Fuente 2 (Countries API) falló: %s. Usando valores por defecto.", e)
        country_info = {
            "pais_nombre_oficial": "Republic of India",
            "capital":    "New Delhi",
            "region":     "Asia",
            "subregion":  "Southern Asia",
            "moneda":     "INR",
            "moneda_nombre": "Indian rupee",
            "idioma_principal": "Hindi",
            "poblacion":  1380004385,
        }

    try:
        exchange_info = extract_exchange_rate()
    except Exception as e:
        logger.warning("Fuente 3 (Exchange API) falló: %s. Usando tasa por defecto.", e)
        exchange_info = {
            "base_currency": "INR",
            "usd_rate":      0.012,
            "last_updated":  "fallback",
        }

    # ── TRANSFORM ─────────────────────────────────────────────────────────────
    logger.info("── FASE TRANSFORM ──")

    try:
        df = limpiar_y_enriquecer(df_raw, exchange_info, country_info)
        df = validar_esquema(df)

        X_cluster    = preparar_features_clustering(df)
        X_pca, pipe_pca = aplicar_pca(X_cluster)
        labels, modelo_km, metricas_km = aplicar_kmeans(X_pca)

        mejor_reg, preprocessor, metricas_sup = entrenar_modelos_supervisados(df)

    except Exception as e:
        logger.critical("Error en fase Transform: %s. Pipeline abortado.", e, exc_info=True)
        sys.exit(1)

    # ── LOAD ──────────────────────────────────────────────────────────────────
    logger.info("── FASE LOAD ──")

    try:
        df["cluster"] = labels
        ruta_csv     = guardar_dataframe(df)
        ruta_labels  = guardar_labels_clusters(df, labels)
        ruta_metrics = guardar_metricas(metricas_km, metricas_sup)
        rutas_modelos = guardar_modelos(pipe_pca, modelo_km, mejor_reg)

        resultados = {
            "archivos": {
                "dataframe":  ruta_csv,
                "labels":     ruta_labels,
                "metricas":   ruta_metrics,
                **rutas_modelos,
            },
            "metricas_clustering": metricas_km,
            "metricas_regresion":  metricas_sup,
        }

    except Exception as e:
        logger.critical("Error en fase Load: %s. Pipeline abortado.", e, exc_info=True)
        sys.exit(1)

    # ── RESUMEN ───────────────────────────────────────────────────────────────
    duracion = round(time.time() - inicio, 2)
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETADO en %.2f segundos", duracion)
    logger.info("Clusters encontrados : %d", metricas_km["n_clusters"])
    logger.info(
        "Silhouette Score     : %.4f", metricas_km["silhouette_score"]
    )
    logger.info(
        "Mejor modelo regresión — R²=%.4f  RMSE=%.4f",
        max(metricas_sup["ridge"]["r2"], metricas_sup["random_forest"]["r2"]),
        min(metricas_sup["ridge"]["rmse"], metricas_sup["random_forest"]["rmse"]),
    )
    logger.info("Archivos generados:")
    for nombre, ruta in resultados["archivos"].items():
        logger.info("  %-20s → %s", nombre, ruta)
    logger.info("=" * 60)

    return resultados


if __name__ == "__main__":
    run_pipeline()
