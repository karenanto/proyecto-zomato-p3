"""
test_etl.py
===========
Tests automatizados para el pipeline ETL Zomato.

Cubre:
    - Carga correcta del CSV principal
    - Validación de columnas requeridas
    - Rangos válidos de variables clave
    - Ausencia de nulos en columnas críticas tras transformación
    - Existencia de archivos generados por el pipeline
    - Enriquecimiento con fuentes externas

Uso:
    pytest tests/test_etl.py -v --html=tests/report_etl.html
"""

import json
import os
import sys

import pandas as pd
import pytest

# Agregar raíz al path para importar módulos ETL
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

PROCESSED_PATH = os.getenv("DATA_PROCESSED_PATH", "data/processed/")
RAW_PATH       = os.getenv("DATA_RAW_PATH",       "data/raw/")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def df_raw():
    """Carga el CSV original para tests de extracción."""
    path = os.path.join(RAW_PATH, "zomato_clean.csv")
    if not os.path.exists(path):
        pytest.skip(f"CSV original no encontrado: {path}")
    return pd.read_csv(path, on_bad_lines="skip")


@pytest.fixture(scope="module")
def df_procesado():
    """Carga el CSV procesado generado por el pipeline."""
    path = os.path.join(PROCESSED_PATH, "zomato_final.csv")
    if not os.path.exists(path):
        pytest.skip("Pipeline no ejecutado aún. Corre python etl/pipeline.py primero.")
    return pd.read_csv(path, low_memory=False)


@pytest.fixture(scope="module")
def metricas():
    """Carga el JSON de métricas generado por el pipeline."""
    path = os.path.join(PROCESSED_PATH, "model_metrics.json")
    if not os.path.exists(path):
        pytest.skip("model_metrics.json no encontrado.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Tests: Extracción ─────────────────────────────────────────────────────────

class TestExtraccion:
    def test_csv_carga_correctamente(self, df_raw):
        """El CSV principal debe cargarse sin errores y tener filas."""
        assert df_raw is not None
        assert len(df_raw) > 0, "El CSV está vacío"

    def test_columnas_requeridas_existen(self, df_raw):
        """El CSV debe contener las columnas mínimas requeridas."""
        columnas_requeridas = {"rate", "votes", "approx_cost(for two people)"}
        faltantes = columnas_requeridas - set(df_raw.columns)
        assert not faltantes, f"Columnas faltantes: {faltantes}"

    def test_csv_tiene_suficientes_filas(self, df_raw):
        """El dataset debe tener al menos 1000 filas para ser útil."""
        assert len(df_raw) >= 1000, f"Solo {len(df_raw)} filas, se esperaban al menos 1000"


# ── Tests: Transformación ─────────────────────────────────────────────────────

class TestTransformacion:
    def test_columna_cluster_existe(self, df_procesado):
        """El pipeline debe agregar una columna 'cluster'."""
        assert "cluster" in df_procesado.columns, "Columna 'cluster' no encontrada"

    def test_columna_costo_usd_existe(self, df_procesado):
        """El enriquecimiento con Exchange Rate debe crear 'costo_usd'."""
        assert "costo_usd" in df_procesado.columns, "Columna 'costo_usd' no encontrada"

    def test_columna_pais_region_existe(self, df_procesado):
        """El enriquecimiento con Countries API debe crear 'pais_region'."""
        assert "pais_region" in df_procesado.columns, "Columna 'pais_region' no encontrada"

    def test_rate_en_rango_valido(self, df_procesado):
        """Los ratings deben estar entre 1.0 y 5.0."""
        rates = df_procesado["rate"].dropna()
        assert (rates >= 1.0).all(), "Hay ratings menores a 1.0"
        assert (rates <= 5.0).all(), "Hay ratings mayores a 5.0"

    def test_votes_no_negativos(self, df_procesado):
        """Los votos no pueden ser negativos."""
        votes = df_procesado["votes"].dropna()
        assert (votes >= 0).all(), "Hay votos negativos"

    def test_costo_usd_no_negativo(self, df_procesado):
        """El costo en USD no puede ser negativo."""
        costos = df_procesado["costo_usd"].dropna()
        assert (costos >= 0).all(), "Hay costos en USD negativos"

    def test_sin_nulos_en_cluster(self, df_procesado):
        """Todos los registros deben tener un cluster asignado."""
        nulos = df_procesado["cluster"].isnull().sum()
        assert nulos == 0, f"Hay {nulos} registros sin cluster asignado"

    def test_numero_clusters_valido(self, df_procesado):
        """El número de clusters debe estar entre 2 y 10."""
        n_clusters = df_procesado["cluster"].nunique()
        assert 2 <= n_clusters <= 10, f"Número de clusters inválido: {n_clusters}"

    def test_filas_no_se_pierden(self, df_raw, df_procesado):
        """El dataset procesado no debe perder más del 5% de las filas originales."""
        perdida = (len(df_raw) - len(df_procesado)) / len(df_raw)
        assert perdida <= 0.05, f"Se perdió el {perdida*100:.1f}% de las filas"


# ── Tests: Carga (archivos generados) ─────────────────────────────────────────

class TestCarga:
    def test_zomato_final_existe(self):
        """El archivo zomato_final.csv debe existir tras el pipeline."""
        path = os.path.join(PROCESSED_PATH, "zomato_final.csv")
        assert os.path.exists(path), f"No se encontró: {path}"

    def test_cluster_labels_existe(self):
        """El archivo cluster_labels.csv debe existir."""
        path = os.path.join(PROCESSED_PATH, "cluster_labels.csv")
        assert os.path.exists(path), f"No se encontró: {path}"

    def test_model_metrics_existe(self):
        """El archivo model_metrics.json debe existir."""
        path = os.path.join(PROCESSED_PATH, "model_metrics.json")
        assert os.path.exists(path), f"No se encontró: {path}"

    def test_modelos_pkl_existen(self):
        """Los modelos serializados deben existir."""
        for nombre in ["pipeline_pca.pkl", "modelo_kmeans.pkl", "modelo_regresion.pkl"]:
            path = os.path.join(PROCESSED_PATH, nombre)
            assert os.path.exists(path), f"No se encontró: {path}"

    def test_log_etl_existe(self):
        """El log del pipeline debe existir."""
        path = os.path.join(PROCESSED_PATH, "etl.log")
        assert os.path.exists(path), f"No se encontró: {path}"


# ── Tests: Métricas ───────────────────────────────────────────────────────────

class TestMetricas:
    def test_metricas_clustering_existen(self, metricas):
        """El JSON de métricas debe contener la sección 'clustering'."""
        assert "clustering" in metricas, "Sección 'clustering' no encontrada en métricas"

    def test_metricas_regresion_existen(self, metricas):
        """El JSON de métricas debe contener la sección 'regresion'."""
        assert "regresion" in metricas, "Sección 'regresion' no encontrada en métricas"

    def test_silhouette_en_rango(self, metricas):
        """El Silhouette Score debe estar entre -1 y 1."""
        score = metricas["clustering"].get("silhouette_score", None)
        assert score is not None, "Silhouette Score no encontrado"
        assert -1 <= score <= 1, f"Silhouette Score fuera de rango: {score}"

    def test_r2_ridge_razonable(self, metricas):
        """El R² de Ridge no debe ser negativo (modelo peor que la media)."""
        r2 = metricas["regresion"]["ridge"].get("r2", None)
        assert r2 is not None, "R² de Ridge no encontrado"
        assert r2 >= 0, f"R² de Ridge negativo: {r2}"

    def test_r2_rf_razonable(self, metricas):
        """El R² de Random Forest no debe ser negativo."""
        r2 = metricas["regresion"]["random_forest"].get("r2", None)
        assert r2 is not None, "R² de Random Forest no encontrado"
        assert r2 >= 0, f"R² de Random Forest negativo: {r2}"
