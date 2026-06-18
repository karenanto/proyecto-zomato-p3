"""
main.py
=======
API RESTful Zomato Analytics — construida con FastAPI.

Endpoints disponibles:
    GET  /                      — Estado de la API
    GET  /clusters              — Perfil de cada cluster
    GET  /metricas/clustering   — Métricas del modelo K-Means
    GET  /metricas/regresion    — Métricas de Ridge y Random Forest
    POST /predecir/rate         — Predice el rating de un restaurante
    POST /predecir/cluster      — Predice a qué cluster pertenece

Uso:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Documentación automática:
    http://localhost:8000/docs
"""

import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()
logger = logging.getLogger(__name__)

# ── Configuración ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PROCESSED = os.path.join(BASE_DIR, "data", "processed")

PROCESSED_PATH = os.getenv("DATA_PROCESSED_PATH", DEFAULT_PROCESSED)

app = FastAPI(
    title="Zomato Analytics API",
    description="API para consultar resultados del pipeline ETL: clusters, métricas y predicciones.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Carga de modelos y datos al iniciar ───────────────────────────────────────
def _cargar_recursos():
    """Carga modelos, métricas y dataset procesado desde disco."""
    recursos = {}

    try:
        recursos["df"] = pd.read_csv(
            os.path.join(PROCESSED_PATH, "zomato_final.csv"), low_memory=False
        )
        logger.info("Dataset cargado: %d filas", len(recursos["df"]))
    except FileNotFoundError:
        logger.error("zomato_final.csv no encontrado. Corre el pipeline ETL primero.")
        recursos["df"] = None

    try:
        with open(os.path.join(PROCESSED_PATH, "model_metrics.json"), encoding="utf-8") as f:
            recursos["metricas"] = json.load(f)
    except FileNotFoundError:
        logger.error("model_metrics.json no encontrado.")
        recursos["metricas"] = {}

    try:
        recursos["pipeline_pca"]     = joblib.load(os.path.join(PROCESSED_PATH, "pipeline_pca.pkl"))
        recursos["modelo_kmeans"]    = joblib.load(os.path.join(PROCESSED_PATH, "modelo_kmeans.pkl"))
        recursos["modelo_regresion"] = joblib.load(os.path.join(PROCESSED_PATH, "modelo_regresion.pkl"))
        logger.info("Modelos cargados correctamente.")
    except FileNotFoundError as e:
        logger.error("Modelo no encontrado: %s", e)
        recursos["pipeline_pca"]     = None
        recursos["modelo_kmeans"]    = None
        recursos["modelo_regresion"] = None

    return recursos


RECURSOS = _cargar_recursos()


# ── Modelos Pydantic (esquemas de entrada/salida) ─────────────────────────────

class EntradaPrediccionRate(BaseModel):
    """Datos de un restaurante para predecir su rating."""
    votes:                        int   = Field(..., ge=0,    example=500,   description="Número de votos")
    approx_cost_for_two:          float = Field(..., ge=0,    example=600.0, description="Costo aprox. para dos (INR)")
    online_order:                 bool  = Field(...,          example=True,  description="¿Acepta pedidos online?")
    book_table:                   bool  = Field(...,          example=False, description="¿Permite reserva de mesa?")
    location:                     str   = Field(...,          example="Koramangala 5th Block", description="Ubicación del restaurante")
    listed_in_type:               str   = Field(...,          example="Delivery", description="Tipo de servicio")


class EntradaPrediccionCluster(BaseModel):
    """Datos numéricos de un restaurante para predecir su cluster."""
    votes:               int   = Field(..., ge=0,  example=500)
    approx_cost_for_two: float = Field(..., ge=0,  example=600.0)
    costo_usd:           float = Field(..., ge=0,  example=7.2)


class RespuestaCluster(BaseModel):
    cluster:     int
    descripcion: str


class RespuestaRate(BaseModel):
    rate_predicho: float
    interpretacion: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _interpretar_rate(rate: float) -> str:
    if rate >= 4.5:
        return "Excelente"
    elif rate >= 4.0:
        return "Muy bueno"
    elif rate >= 3.5:
        return "Bueno"
    elif rate >= 3.0:
        return "Regular"
    else:
        return "Por debajo del promedio"


def _describir_cluster(cluster_id: int, df: pd.DataFrame) -> str:
    """Genera una descripción en lenguaje de negocio para cada cluster."""
    subset = df[df["cluster"] == cluster_id]
    if subset.empty:
        return "Sin datos suficientes"

    avg_rate  = subset["rate"].mean() if "rate" in subset else 0
    avg_votes = subset["votes"].mean() if "votes" in subset else 0
    avg_cost  = subset["approx_cost(for two people)"].mean() if "approx_cost(for two people)" in subset else 0

    rating_label = "alta calificación" if avg_rate >= 4.0 else "calificación media" if avg_rate >= 3.5 else "calificación baja"
    demand_label = "alta demanda" if avg_votes >= 500 else "demanda moderada" if avg_votes >= 200 else "baja demanda"
    cost_label   = "premium" if avg_cost >= 800 else "precio medio" if avg_cost >= 400 else "económico"

    return (
        f"Restaurantes de {cost_label} con {rating_label} y {demand_label}. "
        f"Rating promedio: {avg_rate:.2f} | Votos promedio: {avg_votes:.0f} | "
        f"Costo promedio para dos: ₹{avg_cost:.0f}"
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Estado"])
def root():
    """Verifica que la API está activa."""
    return {
        "status":  "online",
        "version": "1.0.0",
        "mensaje": "Zomato Analytics API funcionando correctamente.",
        "docs":    "/docs",
    }


@app.get("/clusters", tags=["Clustering"])
def get_clusters():
    """
    Retorna el perfil de cada cluster encontrado por K-Means.

    Incluye métricas promedio (rating, votos, costo) y una descripción
    en lenguaje de negocio para cada segmento.
    """
    df = RECURSOS.get("df")
    if df is None or "cluster" not in df.columns:
        raise HTTPException(
            status_code=503,
            detail="Dataset no disponible. Ejecuta el pipeline ETL primero.",
        )

    clusters = []
    for cluster_id in sorted(df["cluster"].unique()):
        subset = df[df["cluster"] == cluster_id]
        clusters.append({
            "cluster_id":    int(cluster_id),
            "n_restaurantes": int(len(subset)),
            "descripcion":   _describir_cluster(cluster_id, df),
            "promedios": {
                "rate":     round(float(subset["rate"].mean()), 3) if "rate" in subset else None,
                "votes":    round(float(subset["votes"].mean()), 1) if "votes" in subset else None,
                "costo_inr": round(float(subset["approx_cost(for two people)"].mean()), 1)
                             if "approx_cost(for two people)" in subset else None,
                "costo_usd": round(float(subset["costo_usd"].mean()), 4)
                             if "costo_usd" in subset else None,
            },
        })

    return {"total_clusters": len(clusters), "clusters": clusters}


@app.get("/metricas/clustering", tags=["Métricas"])
def get_metricas_clustering():
    """
    Retorna las métricas del modelo de clustering K-Means.

    Incluye Silhouette Score, Calinski-Harabasz, Davies-Bouldin e Inercia.
    """
    metricas = RECURSOS.get("metricas", {})
    clustering = metricas.get("clustering")
    if not clustering:
        raise HTTPException(status_code=503, detail="Métricas no disponibles.")
    return clustering


@app.get("/metricas/regresion", tags=["Métricas"])
def get_metricas_regresion():
    """
    Retorna las métricas de los modelos de regresión (Ridge y Random Forest).

    Incluye RMSE, MAE y R² para cada modelo.
    """
    metricas = RECURSOS.get("metricas", {})
    regresion = metricas.get("regresion")
    if not regresion:
        raise HTTPException(status_code=503, detail="Métricas no disponibles.")
    return regresion


@app.post("/predecir/rate", response_model=RespuestaRate, tags=["Predicción"])
def predecir_rate(entrada: EntradaPrediccionRate):
    """
    Predice el rating de un restaurante según sus características.

    Usa el mejor modelo de regresión entrenado (Ridge o Random Forest).
    """
    modelo = RECURSOS.get("modelo_regresion")
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo de regresión no disponible.")

    try:
        df_entrada = pd.DataFrame([{
            "online_order_Yes":              int(entrada.online_order),
            "book_table_Yes":                int(entrada.book_table),
            "approx_cost(for two people)":   entrada.approx_cost_for_two,
            "votes":                         entrada.votes,
            "costo_usd":                     round(entrada.approx_cost_for_two * 0.012, 4),
            "location":                      entrada.location,
            "listed_in(type)":               entrada.listed_in_type,
        }])

        rate_pred = float(modelo.predict(df_entrada)[0])
        rate_pred = round(max(1.0, min(5.0, rate_pred)), 2)

        return RespuestaRate(
            rate_predicho=rate_pred,
            interpretacion=_interpretar_rate(rate_pred),
        )

    except Exception as e:
        logger.error("Error en predicción de rate: %s", e)
        raise HTTPException(status_code=500, detail=f"Error al predecir: {str(e)}")


@app.post("/predecir/cluster", response_model=RespuestaCluster, tags=["Predicción"])
def predecir_cluster(entrada: EntradaPrediccionCluster):
    """
    Predice a qué cluster pertenece un restaurante según sus features numéricas.

    Aplica el mismo pipeline PCA + K-Means usado en el entrenamiento.
    """
    pipeline_pca  = RECURSOS.get("pipeline_pca")
    modelo_kmeans = RECURSOS.get("modelo_kmeans")
    df            = RECURSOS.get("df")

    if pipeline_pca is None or modelo_kmeans is None:
        raise HTTPException(status_code=503, detail="Modelos de clustering no disponibles.")

    try:
        X = np.array([[entrada.votes, entrada.approx_cost_for_two, entrada.costo_usd]])
        X_pca    = pipeline_pca.transform(X)
        cluster  = int(modelo_kmeans.predict(X_pca)[0])
        descripcion = _describir_cluster(cluster, df) if df is not None else "Sin descripción"

        return RespuestaCluster(cluster=cluster, descripcion=descripcion)

    except Exception as e:
        logger.error("Error en predicción de cluster: %s", e)
        raise HTTPException(status_code=500, detail=f"Error al predecir: {str(e)}")
