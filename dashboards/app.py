"""
app.py
======
Dashboard interactivo Zomato Analytics — construido con Streamlit.

Dos vistas diferenciadas por audiencia:
    - Tab Negocio:  perfiles de segmentos, top features, simulador de rating
    - Tab Técnico:  PCA, métricas de clustering, comparativa de modelos

Uso:
    streamlit run dashboards/app.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

load_dotenv()

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Zomato Analytics",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PROCESSED = os.path.join(BASE_DIR, "data", "processed")

PROCESSED_PATH = os.getenv("DATA_PROCESSED_PATH", DEFAULT_PROCESSED)
API_URL        = os.getenv("API_URL", "http://localhost:8000")

# ── Carga de datos ────────────────────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    path = os.path.join(PROCESSED_PATH, "zomato_final.csv")
    try:
        return pd.read_csv(path, low_memory=False)
    except FileNotFoundError:
        st.error("❌ No se encontró zomato_final.csv. Ejecuta el pipeline ETL primero.")
        return None

@st.cache_data
def cargar_metricas():
    path = os.path.join(PROCESSED_PATH, "model_metrics.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

@st.cache_resource
def cargar_modelos():
    try:
        pipeline_pca  = joblib.load(os.path.join(PROCESSED_PATH, "pipeline_pca.pkl"))
        modelo_km     = joblib.load(os.path.join(PROCESSED_PATH, "modelo_kmeans.pkl"))
        modelo_reg    = joblib.load(os.path.join(PROCESSED_PATH, "modelo_regresion.pkl"))
        return pipeline_pca, modelo_km, modelo_reg
    except FileNotFoundError:
        return None, None, None

df                               = cargar_datos()
metricas                         = cargar_metricas()
pipeline_pca, modelo_km, modelo_reg = cargar_modelos()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🍽️ Zomato Analytics")
    st.markdown("---")
    st.markdown("**Fuentes de datos**")
    st.markdown("- 📄 Dataset Zomato (CSV)")
    st.markdown("- 🌍 Rest Countries API")
    st.markdown("- 💱 Open Exchange Rates API")
    st.markdown("---")
    if df is not None:
        st.metric("Total restaurantes", f"{len(df):,}")
        if "cluster" in df.columns:
            st.metric("Clusters encontrados", df["cluster"].nunique())
        if "rate" in df.columns:
            st.metric("Rating promedio", f"{df['rate'].mean():.2f} ⭐")
    st.markdown("---")
    st.caption("Parcial 3 — Ciencia de Datos")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_negocio, tab_tecnico = st.tabs(["📊 Vista de Negocio", "🔬 Vista Técnica"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB NEGOCIO
# ══════════════════════════════════════════════════════════════════════════════
with tab_negocio:
    st.header("🍽️ Análisis de Restaurantes Zomato")

    if df is None:
        st.stop()

    DESCRIPCIONES = {
        0: "🟢 Segmento Económico Popular",
        1: "🔵 Segmento Precio Medio",
        2: "🟡 Segmento Premium",
        3: "🔴 Segmento Alta Gama",
        4: "🟣 Segmento Especializado",
    }

    # Perfil de segmentos
    st.subheader("🎯 Segmentos de Restaurantes")
    if "cluster" in df.columns:
        cols = st.columns(min(df["cluster"].nunique(), 4))
        for i, cluster_id in enumerate(sorted(df["cluster"].unique())):
            subset = df[df["cluster"] == cluster_id]
            label  = DESCRIPCIONES.get(cluster_id, f"Segmento {cluster_id}")
            with cols[i % len(cols)]:
                st.metric(label, f"{len(subset):,} restaurantes")
                if "rate" in df.columns:
                    st.caption(f"⭐ Rating promedio: {subset['rate'].mean():.2f}")
                if "votes" in df.columns:
                    st.caption(f"👥 Votos promedio: {subset['votes'].mean():.0f}")
                if "approx_cost(for two people)" in df.columns:
                    st.caption(f"💰 Costo promedio: ₹{subset['approx_cost(for two people)'].mean():.0f}")

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            conteo = df["cluster"].value_counts().reset_index()
            conteo.columns = ["Cluster", "Restaurantes"]
            conteo["Segmento"] = conteo["Cluster"].map(
                lambda x: DESCRIPCIONES.get(x, f"Segmento {x}")
            )
            fig_pie = px.pie(
                conteo, values="Restaurantes", names="Segmento",
                title="Distribución por Segmento",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            if "rate" in df.columns and "approx_cost(for two people)" in df.columns:
                perfil = df.groupby("cluster").agg(
                    rating=("rate", "mean"),
                    costo=("approx_cost(for two people)", "mean"),
                ).reset_index()
                perfil["Segmento"] = perfil["cluster"].map(
                    lambda x: DESCRIPCIONES.get(x, f"Segmento {x}")
                )
                fig_bar = px.bar(
                    perfil, x="Segmento", y="rating",
                    title="Rating Promedio por Segmento",
                    color="Segmento",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    text_auto=".2f",
                )
                fig_bar.update_layout(showlegend=False, xaxis_tickangle=-20)
                st.plotly_chart(fig_bar, use_container_width=True)

    # Explorador
    st.markdown("---")
    st.subheader("🔍 Explorador de Restaurantes")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        if "cluster" in df.columns:
            cluster_sel = st.selectbox(
                "Filtrar por segmento",
                options=["Todos"] + sorted(df["cluster"].unique().tolist()),
                format_func=lambda x: "Todos" if x == "Todos" else DESCRIPCIONES.get(x, f"Segmento {x}"),
            )
    with col_f2:
        if "rate" in df.columns:
            rating_min = st.slider("Rating mínimo", 1.0, 5.0, 3.5, 0.1)

    df_filtrado = df.copy()
    if "cluster" in df.columns and cluster_sel != "Todos":
        df_filtrado = df_filtrado[df_filtrado["cluster"] == cluster_sel]
    if "rate" in df.columns:
        df_filtrado = df_filtrado[df_filtrado["rate"] >= rating_min]

    cols_mostrar = [c for c in ["name", "location", "rate", "votes",
                                "approx_cost(for two people)", "cluster"]
                    if c in df_filtrado.columns]
    st.dataframe(df_filtrado[cols_mostrar].head(50), use_container_width=True)
    st.caption(f"Mostrando {min(50, len(df_filtrado))} de {len(df_filtrado):,} restaurantes")

    # Simulador
    st.markdown("---")
    st.subheader("🎮 Simulador de Rating")
    with st.form("form_simulador"):
        col_a, col_b = st.columns(2)
        with col_a:
            votos   = st.number_input("Número de votos", min_value=0, value=300)
            costo   = st.number_input("Costo para dos (₹ INR)", min_value=0.0, value=500.0)
            online  = st.checkbox("¿Acepta pedidos online?", value=True)
            reserva = st.checkbox("¿Permite reserva de mesa?", value=False)
        with col_b:
            ubicaciones = sorted(df["location"].dropna().unique().tolist()) if "location" in df.columns else ["Koramangala"]
            location    = st.selectbox("Ubicación", ubicaciones)
            tipos       = sorted(df["listed_in(type)"].dropna().unique().tolist()) if "listed_in(type)" in df.columns else ["Delivery"]
            tipo        = st.selectbox("Tipo de servicio", tipos)
        submitted = st.form_submit_button("🔮 Predecir Rating", use_container_width=True)

    if submitted:
        try:
            payload = {
                "votes":               votos,
                "approx_cost_for_two": costo,
                "online_order":        online,
                "book_table":          reserva,
                "location":            location,
                "listed_in_type":      tipo,
            }
            resp = requests.post(f"{API_URL}/predecir/rate", json=payload, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    st.metric("⭐ Rating Predicho", f"{data['rate_predicho']:.2f} / 5.0")
                with col_r2:
                    st.metric("📊 Interpretación", data["interpretacion"])
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=data["rate_predicho"],
                    gauge={
                        "axis": {"range": [1, 5]},
                        "bar":  {"color": "#2ecc71"},
                        "steps": [
                            {"range": [1, 3],   "color": "#e74c3c"},
                            {"range": [3, 3.5], "color": "#f39c12"},
                            {"range": [3.5, 4], "color": "#f1c40f"},
                            {"range": [4, 5],   "color": "#2ecc71"},
                        ],
                    },
                    title={"text": "Rating Predicho"},
                ))
                st.plotly_chart(fig_gauge, use_container_width=True)
            else:
                st.error(f"Error de la API: {resp.text}")
        except requests.ConnectionError:
            st.warning("⚠️ La API no está disponible. Asegúrate de que esté corriendo en el puerto 8000.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB TÉCNICO
# ══════════════════════════════════════════════════════════════════════════════
with tab_tecnico:
    st.header("🔬 Análisis Técnico de Modelos")

    if df is None:
        st.stop()

    # PCA
    st.subheader("📉 Visualización PCA — Clusters K-Means")
    if "cluster" in df.columns:
        try:
            excluir = [
                "rate", "name", "location", "rest_type", "cuisines",
                "listed_in(type)", "listed_in(city)", "online_order",
                "book_table", "pais_region", "pais_subregion",
                "pais_moneda", "pais_idioma", "cluster",
            ]
            num_cols  = df.select_dtypes(include="number").columns.tolist()
            feat_cols = [c for c in num_cols if c not in excluir]
            X         = df[feat_cols].fillna(df[feat_cols].median()).values
            pca2      = PCA(n_components=2, random_state=42)
            X_2d      = pca2.fit_transform(StandardScaler().fit_transform(X))
            n         = min(3000, len(df))
            idx       = np.random.RandomState(42).choice(len(df), n, replace=False)
            df_pca    = pd.DataFrame({
                "PC1":     X_2d[idx, 0],
                "PC2":     X_2d[idx, 1],
                "Cluster": df["cluster"].iloc[idx].astype(str).values,
            })
            fig_pca = px.scatter(
                df_pca, x="PC1", y="PC2", color="Cluster",
                title="Proyección PCA 2D — Clusters K-Means",
                color_discrete_sequence=px.colors.qualitative.Set1,
                opacity=0.6,
            )
            fig_pca.update_traces(marker=dict(size=4))
            st.plotly_chart(fig_pca, use_container_width=True)
            var_exp = pca2.explained_variance_ratio_
            st.caption(
                f"PC1 explica {var_exp[0]*100:.1f}% | "
                f"PC2 explica {var_exp[1]*100:.1f}% | "
                f"Total: {sum(var_exp)*100:.1f}%"
            )
        except Exception as e:
            st.warning(f"No se pudo graficar PCA: {e}")

    # Métricas clustering
    st.markdown("---")
    st.subheader("📐 Métricas de Clustering")
    m_clust = metricas.get("clustering", {})
    if m_clust:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Silhouette Score",  f"{m_clust.get('silhouette_score', 'N/A')}")
        col2.metric("Calinski-Harabasz", f"{m_clust.get('calinski_harabasz', 'N/A')}")
        col3.metric("Davies-Bouldin",    f"{m_clust.get('davies_bouldin', 'N/A')}")
        col4.metric("Clusters",          f"{m_clust.get('n_clusters', 'N/A')}")
        with st.expander("¿Cómo interpretar estas métricas?"):
            st.markdown("""
| Métrica | Mejor valor | Qué mide |
|---|---|---|
| **Silhouette Score** | Cercano a 1 | Separación entre clusters |
| **Calinski-Harabasz** | Más alto mejor | Densidad y separación |
| **Davies-Bouldin** | Cercano a 0 | Similitud entre clusters |
""")

    # Métricas regresión
    st.markdown("---")
    st.subheader("📈 Comparativa de Modelos de Regresión")
    m_reg = metricas.get("regresion", {})
    if m_reg:
        ridge = m_reg.get("ridge", {})
        rf    = m_reg.get("random_forest", {})
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Ridge Regression**")
            st.metric("R²",   ridge.get("r2", "N/A"))
            st.metric("RMSE", ridge.get("rmse", "N/A"))
            st.metric("MAE",  ridge.get("mae", "N/A"))
        with col2:
            st.markdown("**Random Forest**")
            st.metric("R²",   rf.get("r2", "N/A"))
            st.metric("RMSE", rf.get("rmse", "N/A"))
            st.metric("MAE",  rf.get("mae", "N/A"))
        fig_comp = go.Figure()
        modelos   = ["Ridge Regression", "Random Forest"]
        r2_vals   = [ridge.get("r2", 0), rf.get("r2", 0)]
        rmse_vals = [ridge.get("rmse", 0), rf.get("rmse", 0)]
        fig_comp.add_trace(go.Bar(name="R²",   x=modelos, y=r2_vals,   marker_color=["#3498db", "#2ecc71"]))
        fig_comp.add_trace(go.Bar(name="RMSE", x=modelos, y=rmse_vals, marker_color=["#e74c3c", "#f39c12"]))
        fig_comp.update_layout(title="Comparativa R² y RMSE", barmode="group", yaxis_title="Valor")
        st.plotly_chart(fig_comp, use_container_width=True)

    # Distribuciones
    st.markdown("---")
    st.subheader("📊 Distribuciones de Variables Clave")
    col1, col2 = st.columns(2)
    with col1:
        if "rate" in df.columns:
            fig_hist = px.histogram(
                df.dropna(subset=["rate"]), x="rate",
                nbins=30, title="Distribución de Ratings",
                color_discrete_sequence=["#3498db"],
            )
            st.plotly_chart(fig_hist, use_container_width=True)
    with col2:
        if "votes" in df.columns:
            fig_votes = px.histogram(
                df[df["votes"] < df["votes"].quantile(0.95)],
                x="votes", nbins=30,
                title="Distribución de Votos (sin outliers)",
                color_discrete_sequence=["#e67e22"],
            )
            st.plotly_chart(fig_votes, use_container_width=True)

    if "rate" in df.columns and "approx_cost(for two people)" in df.columns and "cluster" in df.columns:
        n      = min(2000, len(df))
        idx    = np.random.RandomState(42).choice(len(df), n, replace=False)
        df_sc  = df.iloc[idx].dropna(subset=["rate", "approx_cost(for two people)"])
        fig_sc = px.scatter(
            df_sc,
            x="approx_cost(for two people)", y="rate",
            color=df_sc["cluster"].astype(str),
            title="Rating vs Costo para dos personas — por Cluster",
            labels={"approx_cost(for two people)": "Costo (₹ INR)", "rate": "Rating"},
            opacity=0.5,
            color_discrete_sequence=px.colors.qualitative.Set1,
        )
        st.plotly_chart(fig_sc, use_container_width=True)
