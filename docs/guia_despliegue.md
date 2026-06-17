# Guía de Despliegue — Zomato Analytics

## Requisitos previos

| Herramienta | Versión mínima | Verificar con |
|---|---|---|
| Python | 3.11 | `python --version` |
| Git | 2.x | `git --version` |
| Docker Desktop | 4.x | `docker --version` |

---

## Opción A — Despliegue local (sin Docker)

### 1. Clonar el repositorio

```bash
git clone https://github.com/TU_USUARIO/proyecto-zomato-p3.git
cd proyecto-zomato-p3
```

### 2. Instalar dependencias

```bash
pip install pandas numpy scikit-learn pandera requests joblib python-dotenv
pip install fastapi uvicorn
pip install streamlit plotly
pip install pytest pytest-html httpx
```

### 3. Configurar variables de entorno

```bash
cp .env.example .env
```

El archivo `.env` por defecto funciona sin modificaciones. Si necesitas cambiar puertos o rutas, edítalo antes de continuar.

### 4. Agregar el dataset

Copia `zomato_clean.csv` a la carpeta `data/raw/`:

```bash
# Windows
copy ruta\a\zomato_clean.csv data\raw\zomato_clean.csv

# Mac/Linux
cp ruta/a/zomato_clean.csv data/raw/zomato_clean.csv
```

### 5. Ejecutar el pipeline ETL

```bash
python etl/pipeline.py
```

Espera hasta ver `PIPELINE COMPLETADO` en la consola. Esto genera todos los archivos en `data/processed/`.

### 6. Iniciar la API

En una terminal:

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Verifica que esté activa en: http://localhost:8000

### 7. Iniciar el dashboard

En otra terminal:

```bash
python -m streamlit run dashboards/app.py
```

Se abre automáticamente en: http://localhost:8501

---

## Opción B — Despliegue con Docker

### 1. Clonar y configurar

```bash
git clone https://github.com/TU_USUARIO/proyecto-zomato-p3.git
cd proyecto-zomato-p3
cp .env.example .env
cp .env docker/.env
```

### 2. Agregar el dataset

```bash
# Copiar zomato_clean.csv a data/raw/
```

### 3. Abrir Docker Desktop

Espera hasta que el ícono de Docker en la barra de tareas indique que está corriendo.

### 4. Construir y levantar los servicios

```bash
docker-compose up --build
```

Esto construye las 3 imágenes y levanta los servicios en orden:
1. ETL (corre y termina)
2. API en puerto 8000
3. Dashboard en puerto 8501

### 5. Verificar que todo funciona

```bash
# Ver contenedores activos
docker ps

# Ver logs de un servicio específico
docker logs zomato_api
docker logs zomato_dashboard
```

### 6. Detener los servicios

```bash
docker-compose down
```

Para eliminar también los volúmenes de datos:

```bash
docker-compose down -v
```

---

## Correr los tests

Con la API corriendo en el puerto 8000:

```bash
python -m pytest tests/ -v --html=tests/report.html
```

El reporte queda en `tests/report.html`.

---

## Variables de entorno

| Variable | Default | Descripción |
|---|---|---|
| `DATA_RAW_PATH` | `data/raw/` | Ruta de datos originales |
| `DATA_PROCESSED_PATH` | `data/processed/` | Ruta de datos procesados |
| `API_HOST` | `0.0.0.0` | Host de la API |
| `API_PORT` | `8000` | Puerto de la API |
| `API_URL` | `http://localhost:8000` | URL de la API (usada por el dashboard) |
| `DASHBOARD_PORT` | `8501` | Puerto del dashboard |
| `COUNTRIES_API_URL` | `https://restcountries.com/v3.1/name/india` | URL Rest Countries API |
| `EXCHANGE_API_URL` | `https://open.er-api.com/v6/latest/INR` | URL Open Exchange Rates |

---

## Solución de problemas comunes

**`ModuleNotFoundError`**
→ Instala las dependencias desde la raíz: `pip install -r requirements.txt`

**`FileNotFoundError: zomato_clean.csv`**
→ Verifica que el archivo esté en `data/raw/zomato_clean.csv`

**`Address already in use` en puerto 8000**
→ Cambia el puerto: `python -m uvicorn api.main:app --port 8001`

**Dashboard no conecta con la API**
→ Verifica que la API esté corriendo y que `API_URL` en `.env` apunte al puerto correcto

**Docker: `env file not found`**
→ Asegúrate de copiar el `.env` también a la carpeta `docker/`: `cp .env docker/.env`
