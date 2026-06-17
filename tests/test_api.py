"""
test_api.py
===========
Tests automatizados para la API RESTful Zomato Analytics.

Cubre:
    - Estado de la API (GET /)
    - Endpoints de métricas (clustering y regresión)
    - Endpoint de clusters (GET /clusters)
    - Predicción de rating con payload válido (POST /predecir/rate)
    - Predicción de rating con payload inválido (error 422)
    - Predicción de cluster (POST /predecir/cluster)

Requisito:
    La API debe estar corriendo en API_URL (default: http://localhost:8000)

Uso:
    pytest tests/test_api.py -v --html=tests/report_api.html
"""

import os
import pytest
import httpx

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── Fixture: cliente HTTP ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cliente():
    """Cliente HTTP para la API. Salta tests si la API no está disponible."""
    try:
        resp = httpx.get(f"{API_URL}/", timeout=5)
        resp.raise_for_status()
    except Exception:
        pytest.skip(f"API no disponible en {API_URL}. Asegúrate de que esté corriendo.")
    return httpx.Client(base_url=API_URL, timeout=10)


# ── Tests: Estado ─────────────────────────────────────────────────────────────

class TestEstado:
    def test_raiz_responde_200(self, cliente):
        """GET / debe responder con status 200."""
        resp = cliente.get("/")
        assert resp.status_code == 200

    def test_raiz_contiene_status_online(self, cliente):
        """GET / debe indicar que la API está online."""
        resp = cliente.get("/")
        data = resp.json()
        assert data.get("status") == "online"

    def test_raiz_contiene_version(self, cliente):
        """GET / debe incluir el campo 'version'."""
        resp = cliente.get("/")
        assert "version" in resp.json()


# ── Tests: Clusters ───────────────────────────────────────────────────────────

class TestClusters:
    def test_clusters_responde_200(self, cliente):
        """GET /clusters debe responder con status 200."""
        resp = cliente.get("/clusters")
        assert resp.status_code == 200

    def test_clusters_contiene_lista(self, cliente):
        """GET /clusters debe retornar una lista de clusters."""
        resp  = cliente.get("/clusters")
        data  = resp.json()
        assert "clusters" in data
        assert isinstance(data["clusters"], list)
        assert len(data["clusters"]) > 0

    def test_clusters_tienen_campos_requeridos(self, cliente):
        """Cada cluster debe tener cluster_id, n_restaurantes y descripcion."""
        resp    = cliente.get("/clusters")
        cluster = resp.json()["clusters"][0]
        assert "cluster_id"      in cluster
        assert "n_restaurantes"  in cluster
        assert "descripcion"     in cluster
        assert "promedios"       in cluster

    def test_total_clusters_es_correcto(self, cliente):
        """total_clusters debe coincidir con el largo de la lista."""
        resp = cliente.get("/clusters")
        data = resp.json()
        assert data["total_clusters"] == len(data["clusters"])


# ── Tests: Métricas ───────────────────────────────────────────────────────────

class TestMetricas:
    def test_metricas_clustering_responde_200(self, cliente):
        """GET /metricas/clustering debe responder con status 200."""
        resp = cliente.get("/metricas/clustering")
        assert resp.status_code == 200

    def test_metricas_clustering_contiene_silhouette(self, cliente):
        """Las métricas de clustering deben incluir silhouette_score."""
        resp = cliente.get("/metricas/clustering")
        assert "silhouette_score" in resp.json()

    def test_metricas_regresion_responde_200(self, cliente):
        """GET /metricas/regresion debe responder con status 200."""
        resp = cliente.get("/metricas/regresion")
        assert resp.status_code == 200

    def test_metricas_regresion_contiene_ridge_y_rf(self, cliente):
        """Las métricas de regresión deben incluir ridge y random_forest."""
        resp = cliente.get("/metricas/regresion")
        data = resp.json()
        assert "ridge"         in data
        assert "random_forest" in data

    def test_metricas_regresion_tienen_r2(self, cliente):
        """Cada modelo de regresión debe tener R², RMSE y MAE."""
        resp = cliente.get("/metricas/regresion")
        data = resp.json()
        for modelo in ["ridge", "random_forest"]:
            assert "r2"   in data[modelo], f"R² faltante en {modelo}"
            assert "rmse" in data[modelo], f"RMSE faltante en {modelo}"
            assert "mae"  in data[modelo], f"MAE faltante en {modelo}"


# ── Tests: Predicción de rating ───────────────────────────────────────────────

class TestPrediccionRate:
    PAYLOAD_VALIDO = {
        "votes":               500,
        "approx_cost_for_two": 600.0,
        "online_order":        True,
        "book_table":          False,
        "location":            "Koramangala 5th Block",
        "listed_in_type":      "Delivery",
    }

    def test_predecir_rate_responde_200(self, cliente):
        """POST /predecir/rate con payload válido debe responder 200."""
        resp = cliente.post("/predecir/rate", json=self.PAYLOAD_VALIDO)
        assert resp.status_code == 200

    def test_predecir_rate_retorna_float(self, cliente):
        """La predicción de rating debe ser un número float."""
        resp = cliente.post("/predecir/rate", json=self.PAYLOAD_VALIDO)
        rate = resp.json().get("rate_predicho")
        assert isinstance(rate, float), f"rate_predicho no es float: {rate}"

    def test_predecir_rate_en_rango(self, cliente):
        """El rating predicho debe estar entre 1.0 y 5.0."""
        resp = cliente.post("/predecir/rate", json=self.PAYLOAD_VALIDO)
        rate = resp.json().get("rate_predicho")
        assert 1.0 <= rate <= 5.0, f"Rating fuera de rango: {rate}"

    def test_predecir_rate_contiene_interpretacion(self, cliente):
        """La respuesta debe incluir una interpretación en texto."""
        resp = cliente.post("/predecir/rate", json=self.PAYLOAD_VALIDO)
        data = resp.json()
        assert "interpretacion" in data
        assert isinstance(data["interpretacion"], str)
        assert len(data["interpretacion"]) > 0

    def test_predecir_rate_payload_invalido_retorna_422(self, cliente):
        """POST /predecir/rate con payload inválido debe retornar 422."""
        payload_invalido = {"votes": "no_es_numero", "approx_cost_for_two": -100}
        resp = cliente.post("/predecir/rate", json=payload_invalido)
        assert resp.status_code == 422

    def test_predecir_rate_votes_negativos_retorna_422(self, cliente):
        """Votos negativos deben ser rechazados con 422."""
        payload = {**self.PAYLOAD_VALIDO, "votes": -1}
        resp = cliente.post("/predecir/rate", json=payload)
        assert resp.status_code == 422


# ── Tests: Predicción de cluster ──────────────────────────────────────────────

class TestPrediccionCluster:
    PAYLOAD_VALIDO = {
        "votes":               500,
        "approx_cost_for_two": 600.0,
        "costo_usd":           7.2,
    }

    def test_predecir_cluster_responde_200(self, cliente):
        """POST /predecir/cluster con payload válido debe responder 200."""
        resp = cliente.post("/predecir/cluster", json=self.PAYLOAD_VALIDO)
        assert resp.status_code == 200

    def test_predecir_cluster_retorna_entero(self, cliente):
        """El cluster predicho debe ser un entero."""
        resp    = cliente.post("/predecir/cluster", json=self.PAYLOAD_VALIDO)
        cluster = resp.json().get("cluster")
        assert isinstance(cluster, int), f"cluster no es entero: {cluster}"

    def test_predecir_cluster_contiene_descripcion(self, cliente):
        """La respuesta debe incluir una descripción del cluster."""
        resp = cliente.post("/predecir/cluster", json=self.PAYLOAD_VALIDO)
        data = resp.json()
        assert "descripcion" in data
        assert isinstance(data["descripcion"], str)

    def test_predecir_cluster_payload_invalido_retorna_422(self, cliente):
        """Payload sin campos requeridos debe retornar 422."""
        resp = cliente.post("/predecir/cluster", json={"votes": 100})
        assert resp.status_code == 422
