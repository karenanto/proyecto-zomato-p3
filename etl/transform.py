"""
transform.py
============
Módulo de transformación del pipeline ETL Zomato.

Convierte las funciones del Parcial 2 en pasos modulares y documentados:
    1. Limpieza y enriquecimiento del DataFrame
    2. Validación de esquema con pandera
    3. Preparación de features para clustering
    4. Reducción dimensional con PCA
    5. Clustering K-Means optimizado
    6. Preparación de features para regresión
    7. Entrenamiento de modelos supervisados
"""

import logging
import warnings
import numpy as np
import pandas as pd
import pandera as pa
import pandera.typing as pat

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    mean_squared_error, mean_absolute_error, r2_score,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42


# ──────────────────────────────────────────────────────────────────────────────
# Paso 1: Limpieza y enriquecimiento
# ──────────────────────────────────────────────────────────────────────────────

def limpiar_y_enriquecer(
    df: pd.DataFrame,
    exchange_info: dict,
    country_info: dict,
) -> pd.DataFrame:
    """
    Limpia el DataFrame principal y lo enriquece con datos de las fuentes 2 y 3.

    Transformaciones aplicadas:
        - Conversión de 'approx_cost' a tipo numérico
        - Creación de columna 'costo_usd' usando la tasa INR→USD
        - Adición de metadatos del país (región, moneda, idioma)
        - Imputación de nulos en columnas numéricas con la mediana

    Args:
        df:            DataFrame crudo del CSV principal.
        exchange_info: Diccionario retornado por extract_exchange_rate().
        country_info:  Diccionario retornado por extract_country_info().

    Returns:
        DataFrame enriquecido y limpio.
    """
    logger.info("Transform — Limpieza y enriquecimiento. Shape inicial: %s", df.shape)
    df = df.copy()

    # Limpiar costo: eliminar comas y convertir a float
    col_costo = "approx_cost(for two people)"
    if col_costo in df.columns:
        df[col_costo] = (
            df[col_costo]
            .astype(str)
            .str.replace(",", "", regex=False)
            .pipe(pd.to_numeric, errors="coerce")
        )
        # Enriquecer: costo en USD (Fuente 3)
        df["costo_usd"] = (df[col_costo] * exchange_info["usd_rate"]).round(4)
        logger.info("Columna 'costo_usd' creada con tasa %.6f", exchange_info["usd_rate"])

    # Enriquecer: metadatos del país (Fuente 2)
    df["pais_region"]   = country_info["region"]
    df["pais_subregion"] = country_info["subregion"]
    df["pais_moneda"]   = country_info["moneda"]
    df["pais_idioma"]   = country_info["idioma_principal"]
    logger.info("Metadatos del país agregados: %s", country_info["pais_nombre_oficial"])

    # Imputar nulos numéricos con la mediana
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    logger.info("Transform — Limpieza OK. Shape final: %s", df.shape)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Paso 2: Validación de esquema con pandera
# ──────────────────────────────────────────────────────────────────────────────

def validar_esquema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida el esquema del DataFrame usando pandera.

    Reglas:
        - 'rate' debe ser float entre 1.0 y 5.0 (nullable)
        - 'votes' debe ser entero >= 0
        - 'approx_cost(for two people)' debe ser float >= 0 (nullable)
        - 'costo_usd' debe ser float >= 0 (nullable)

    Args:
        df: DataFrame a validar.

    Returns:
        El mismo DataFrame si pasa la validación.

    Raises:
        pa.errors.SchemaError: Si alguna columna viola las restricciones.
    """
    logger.info("Transform — Validando esquema con pandera...")

    schema = pa.DataFrameSchema(
        columns={
            "rate": pa.Column(
                float, pa.Check.in_range(1.0, 5.0), nullable=True
            ),
            "votes": pa.Column(
                int, pa.Check.greater_than_or_equal_to(0), nullable=True
            ),
            "approx_cost(for two people)": pa.Column(
                float, pa.Check.greater_than_or_equal_to(0), nullable=True
            ),
            "costo_usd": pa.Column(
                float, pa.Check.greater_than_or_equal_to(0), nullable=True
            ),
        },
        strict=False,   # Permite columnas adicionales
        coerce=True,    # Intenta castear antes de fallar
    )

    try:
        df_validado = schema.validate(df)
        logger.info("Transform — Esquema válido.")
        return df_validado
    except pa.errors.SchemaError as e:
        logger.error("Transform — Error de esquema: %s", e)
        raise


# ──────────────────────────────────────────────────────────────────────────────
# Paso 3: Features para clustering (no supervisado)
# ──────────────────────────────────────────────────────────────────────────────

def preparar_features_clustering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selecciona y prepara las features numéricas para el modelo de clustering.

    Excluye el target 'rate' y columnas de texto original ya encodeadas.
    Incluye 'costo_usd' como feature adicional del enriquecimiento ETL.

    Args:
        df: DataFrame limpio y enriquecido.

    Returns:
        DataFrame con solo las features numéricas para clustering.
    """
    excluir = [
        "rate", "name", "location", "rest_type", "cuisines",
        "listed_in(type)", "listed_in(city)", "online_order", "book_table",
        "pais_region", "pais_subregion", "pais_moneda", "pais_idioma",
    ]
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [c for c in num_cols if c not in excluir]

    logger.info(
        "Transform — Features clustering (%d): %s", len(feature_cols), feature_cols
    )
    X = df[feature_cols].copy().fillna(df[feature_cols].median())
    return X


# ──────────────────────────────────────────────────────────────────────────────
# Paso 4: PCA
# ──────────────────────────────────────────────────────────────────────────────

def aplicar_pca(X: pd.DataFrame, varianza: float = 0.95) -> tuple:
    """
    Aplica escalado estándar y PCA al conjunto de features.

    Args:
        X:        DataFrame de features numéricas.
        varianza: Fracción de varianza a retener (default 0.95).

    Returns:
        Tupla (X_pca, pipeline_pca) donde:
            - X_pca es el array numpy transformado
            - pipeline_pca es el Pipeline sklearn entrenado (para reusar en API)
    """
    logger.info("Transform — Aplicando PCA (varianza=%.2f)...", varianza)

    pipeline_pca = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=varianza, random_state=RANDOM_STATE)),
    ])

    X_pca = pipeline_pca.fit_transform(X)
    n_componentes = pipeline_pca.named_steps["pca"].n_components_

    logger.info(
        "Transform — PCA OK. %d dimensiones → %d componentes.",
        X.shape[1], n_componentes,
    )
    return X_pca, pipeline_pca


# ──────────────────────────────────────────────────────────────────────────────
# Paso 5: K-Means optimizado
# ──────────────────────────────────────────────────────────────────────────────

def _silhouette_scorer(estimator, X):
    """Scorer personalizado para GridSearch: Silhouette Score."""
    labels = estimator.fit_predict(X)
    if len(set(labels)) < 2:
        return -1.0
    return silhouette_score(X, labels, random_state=RANDOM_STATE)


def aplicar_kmeans(X_pca: np.ndarray) -> tuple:
    """
    Entrena K-Means con búsqueda de hiperparámetros (Grid + Randomized).

    Compara ambos métodos y selecciona el de mayor Silhouette Score.

    Args:
        X_pca: Array numpy de datos reducidos con PCA.

    Returns:
        Tupla (labels, modelo, metricas) donde:
            - labels   es el array de etiquetas de cluster por fila
            - modelo   es el KMeans entrenado (para reusar en API)
            - metricas es un dict con Silhouette, CH, DB e Inercia
    """
    logger.info("Transform — Optimizando K-Means...")

    # Espacio reducido para ejecución eficiente
    param_grid = {
        "n_clusters": [3, 4, 5],
        "init":       ["k-means++"],
        "n_init":     [10],
        "max_iter":   [300],
    }
    kmeans_base = KMeans(random_state=RANDOM_STATE)

    grid_search = GridSearchCV(
        kmeans_base, param_grid,
        scoring=_silhouette_scorer, cv=2, n_jobs=1, verbose=0,
    )
    grid_search.fit(X_pca)

    rand_search = RandomizedSearchCV(
        kmeans_base, param_grid,
        n_iter=3, scoring=_silhouette_scorer, cv=2,
        random_state=RANDOM_STATE, n_jobs=1, verbose=0,
    )
    rand_search.fit(X_pca)

    if grid_search.best_score_ >= rand_search.best_score_:
        mejor = grid_search
        metodo = "GridSearchCV"
    else:
        mejor = rand_search
        metodo = "RandomizedSearchCV"

    logger.info(
        "Transform — K-Means ganador: %s (Silhouette=%.4f, params=%s)",
        metodo, mejor.best_score_, mejor.best_params_,
    )

    modelo = mejor.best_estimator_
    labels = modelo.predict(X_pca)

    metricas = {
        "metodo_busqueda":  metodo,
        "silhouette_score": round(silhouette_score(X_pca, labels), 4),
        "calinski_harabasz": round(calinski_harabasz_score(X_pca, labels), 4),
        "davies_bouldin":   round(davies_bouldin_score(X_pca, labels), 4),
        "inercia":          round(modelo.inertia_, 2),
        "n_clusters":       int(len(np.unique(labels))),
        "params":           mejor.best_params_,
    }
    logger.info("Transform — Métricas K-Means: %s", metricas)
    return labels, modelo, metricas


# ──────────────────────────────────────────────────────────────────────────────
# Paso 6 y 7: Regresión supervisada
# ──────────────────────────────────────────────────────────────────────────────

def entrenar_modelos_supervisados(df: pd.DataFrame) -> tuple:
    """
    Entrena Ridge Regression y Random Forest para predecir 'rate'.

    Replica exactamente el pipeline del Parcial 2 (secciones 11–13).

    Args:
        df: DataFrame enriquecido que incluye la columna 'rate'.

    Returns:
        Tupla (mejor_modelo, pipeline_preprocesador, metricas_sup) donde:
            - mejor_modelo         es el modelo con mayor R²
            - pipeline_preprocesador es el ColumnTransformer entrenado
            - metricas_sup         es un dict con RMSE, MAE, R² de ambos modelos
    """
    logger.info("Transform — Entrenando modelos supervisados...")

    features_sup = [
        "online_order_Yes", "book_table_Yes",
        "approx_cost(for two people)", "votes", "costo_usd",
        "location", "listed_in(type)",
    ]
    target = "rate"

    # Filtrar columnas disponibles
    features_disponibles = [f for f in features_sup if f in df.columns]
    df_sup = df[features_disponibles + [target]].dropna()

    X_sup = df_sup[features_disponibles]
    y_sup = df_sup[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X_sup, y_sup, test_size=0.2, random_state=RANDOM_STATE
    )

    numeric_f    = [f for f in features_disponibles
                    if f not in ["location", "listed_in(type)"]]
    categorical_f = [f for f in ["location", "listed_in(type)"]
                     if f in features_disponibles]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_f),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_f),
    ])

    # Ridge
    pipe_ridge = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor",    Ridge()),
    ])
    grid_ridge = GridSearchCV(
        pipe_ridge,
        {"regressor__alpha": [0.1, 1.0, 10.0, 100.0]},
        cv=5, scoring="r2", n_jobs=-1,
    )
    grid_ridge.fit(X_train, y_train)

    # Random Forest
    pipe_rf = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor",    RandomForestRegressor(random_state=RANDOM_STATE)),
    ])
    rand_rf = RandomizedSearchCV(
        pipe_rf,
        {
            "regressor__n_estimators":      [50, 100, 200],
            "regressor__max_depth":         [None, 10, 20],
            "regressor__min_samples_split": [2, 5, 10],
        },
        n_iter=5, cv=3, scoring="r2",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rand_rf.fit(X_train, y_train)

    def _eval(model, name):
        y_pred = model.predict(X_test)
        return {
            "modelo":  name,
            "rmse":    round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
            "mae":     round(float(mean_absolute_error(y_test, y_pred)), 4),
            "r2":      round(float(r2_score(y_test, y_pred)), 4),
            "params":  model.best_params_,
        }

    m_ridge = _eval(grid_ridge, "Ridge Regression")
    m_rf    = _eval(rand_rf,    "Random Forest")

    metricas_sup = {"ridge": m_ridge, "random_forest": m_rf}
    logger.info("Transform — Ridge R²=%.4f | RF R²=%.4f", m_ridge["r2"], m_rf["r2"])

    mejor_modelo = grid_ridge if m_ridge["r2"] >= m_rf["r2"] else rand_rf
    logger.info("Transform — Mejor modelo supervisado: %s", mejor_modelo)

    return mejor_modelo, preprocessor, metricas_sup
