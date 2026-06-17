# 🍽️ Zomato Analytics — Parcial 3

Sistema completo de ingeniería de datos sobre el dataset Zomato, que integra un pipeline ETL con tres fuentes de datos, una API RESTful, un dashboard interactivo y despliegue con Docker.

## Integrantes

- Alonso Díaz
- Amaro Leal
- Karen Manríquez 

## Descripción del proyecto

Este proyecto extiende el análisis del Parcial 2 (clustering K-Means + regresión Ridge/Random Forest sobre restaurantes de India) hacia un sistema de producción completo con las siguientes componentes:

- **Pipeline ETL** que integra 3 fuentes, valida esquemas y genera modelos entrenados
- **API RESTful** con FastAPI que expone métricas y predicciones
- **Dashboard interactivo** con Streamlit diferenciado por audiencia
- **Contenedores Docker** para despliegue reproducible

## Estructura del proyecto

```
proyecto-zomato-p3/
├── etl/                    # Pipeline ETL (extract, transform, load)
├── api/                    # API RESTful FastAPI
├── dashboards/             # Dashboard Streamlit
├── tests/                  # Tests automatizados con pytest
├── docker/                 # Dockerfiles y docker-compose
├── docs/                   # Documentación técnica
├── data/
│   ├── raw/                # Datos originales
│   └── processed/          # Datos y modelos generados
├── .env.example            # Plantilla de variables de entorno
└── requirements.txt        # Dependencias Python
```

## Fuentes de datos

| # | Fuente | Tipo | Qué aporta |
|---|---|---|---|
| 1 | `zomato_clean.csv` | CSV local | Dataset principal de restaurantes |
| 2 | Rest Countries API | HTTP (gratuita) | Metadatos de India: región, moneda, idioma |
| 3 | Open Exchange Rates API | HTTP (gratuita) | Tasa de cambio INR→USD |

## Inicio rápido

### Opción A — Correr localmente

```bash
# 1. Clonar el repositorio
git clone https://github.com/TU_USUARIO/proyecto-zomato-p3.git
cd proyecto-zomato-p3

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar variables de entorno
cp .env.example .env

# 4. Poner el CSV en data/raw/
# (copiar zomato_clean.csv a data/raw/)

# 5. Correr el pipeline ETL
python etl/pipeline.py

# 6. Correr la API (en una terminal)
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# 7. Correr el dashboard (en otra terminal)
python -m streamlit run dashboards/app.py
```

### Opción B — Correr con Docker

```bash
# 1. Clonar y configurar
git clone https://github.com/TU_USUARIO/proyecto-zomato-p3.git
cd proyecto-zomato-p3
cp .env.example .env
cp .env docker/.env

# 2. Levantar todo
docker-compose up --build
```

## URLs de acceso

| Servicio | URL |
|---|---|
| API | http://localhost:8000 |
| Documentación API (Swagger) | http://localhost:8000/docs |
| Dashboard | http://localhost:8501 |

## Tests

```bash
# Asegurarse de que la API esté corriendo, luego:
python -m pytest tests/ -v --html=tests/report.html
```

## Tecnologías utilizadas

- **Python 3.11** — lenguaje principal
- **pandas, scikit-learn, pandera** — ETL y ML
- **FastAPI + Uvicorn** — API RESTful
- **Streamlit + Plotly** — Dashboard
- **Docker + docker-compose** — Contenedores
- **pytest + pytest-html** — Testing
