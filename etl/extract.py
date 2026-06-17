"""
extract.py
==========
Módulo de extracción del pipeline ETL Zomato.

Fuentes:
    1. zomato_clean.csv    — Dataset principal (local)
    2. Rest Countries API  — Datos del país India (HTTP, sin API key)
    3. Open Exchange Rates — Tasa de cambio INR→USD (HTTP, sin API key)
"""

import logging
import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

RAW_PATH = os.getenv("DATA_RAW_PATH", "data/raw/")
COUNTRIES_API_URL = os.getenv(
    "COUNTRIES_API_URL", "https://restcountries.com/v3.1/name/india"
)
EXCHANGE_API_URL = os.getenv(
    "EXCHANGE_API_URL", "https://open.er-api.com/v6/latest/INR"
)


# ──────────────────────────────────────────────────────────────────────────────
# Fuente 1: CSV principal
# ──────────────────────────────────────────────────────────────────────────────

def extract_zomato_csv(filename: str = "zomato_clean.csv") -> pd.DataFrame:
    """
    Carga el dataset principal Zomato desde un archivo CSV local.

    Args:
        filename: Nombre del archivo CSV dentro de DATA_RAW_PATH.

    Returns:
        DataFrame con los datos crudos del CSV.

    Raises:
        FileNotFoundError: Si el archivo no existe en la ruta esperada.
        ValueError: Si el CSV está vacío o no tiene las columnas mínimas.
    """
    path = os.path.join(RAW_PATH, filename)
    logger.info("Fuente 1 — Cargando CSV: %s", path)

    try:
        df = pd.read_csv(path, on_bad_lines="skip")
    except FileNotFoundError:
        logger.error("Archivo no encontrado: %s", path)
        raise

    if df.empty:
        raise ValueError(f"El archivo {path} está vacío.")

    columnas_requeridas = {"rate", "votes", "approx_cost(for two people)"}
    faltantes = columnas_requeridas - set(df.columns)
    if faltantes:
        raise ValueError(f"Columnas faltantes en el CSV: {faltantes}")

    logger.info("Fuente 1 OK — Shape: %s", df.shape)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Fuente 2: Rest Countries API
# ──────────────────────────────────────────────────────────────────────────────

def extract_country_info(timeout: int = 10) -> dict:
    """
    Obtiene metadatos del país India desde la API pública Rest Countries.

    Datos extraídos:
        - Nombre oficial del país
        - Capital
        - Región y subregión
        - Moneda oficial
        - Idiomas oficiales
        - Población

    Args:
        timeout: Segundos máximos de espera para la petición HTTP.

    Returns:
        Diccionario con los campos mencionados.

    Raises:
        requests.HTTPError: Si la API responde con un código de error.
        requests.Timeout: Si la petición supera el tiempo límite.
        KeyError: Si la respuesta no contiene los campos esperados.
    """
    logger.info("Fuente 2 — Consultando Rest Countries API: %s", COUNTRIES_API_URL)

    try:
        response = requests.get(COUNTRIES_API_URL, timeout=timeout)
        response.raise_for_status()
    except requests.Timeout:
        logger.error("Timeout al consultar Rest Countries API.")
        raise
    except requests.HTTPError as e:
        logger.error("Error HTTP en Rest Countries API: %s", e)
        raise

    data = response.json()[0]

    try:
        country_info = {
            "pais_nombre_oficial": data["name"]["official"],
            "capital":             data["capital"][0],
            "region":              data["region"],
            "subregion":           data["subregion"],
            "moneda":              list(data["currencies"].keys())[0],
            "moneda_nombre":       list(data["currencies"].values())[0]["name"],
            "idioma_principal":    list(data["languages"].values())[0],
            "poblacion":           data["population"],
        }
    except KeyError as e:
        logger.error("Campo inesperado en respuesta de Rest Countries: %s", e)
        raise

    logger.info("Fuente 2 OK — País: %s", country_info["pais_nombre_oficial"])
    return country_info


# ──────────────────────────────────────────────────────────────────────────────
# Fuente 3: Open Exchange Rates API
# ──────────────────────────────────────────────────────────────────────────────

def extract_exchange_rate(timeout: int = 10) -> dict:
    """
    Obtiene la tasa de cambio INR→USD desde la API pública Open Exchange Rates.

    Args:
        timeout: Segundos máximos de espera para la petición HTTP.

    Returns:
        Diccionario con:
            - base_currency (str): moneda base (INR)
            - usd_rate (float): cuántos USD equivale 1 INR
            - last_updated (str): fecha de actualización de la tasa

    Raises:
        requests.HTTPError: Si la API responde con un código de error.
        requests.Timeout: Si la petición supera el tiempo límite.
        KeyError: Si la respuesta no contiene los campos esperados.
    """
    logger.info("Fuente 3 — Consultando Open Exchange Rates: %s", EXCHANGE_API_URL)

    try:
        response = requests.get(EXCHANGE_API_URL, timeout=timeout)
        response.raise_for_status()
    except requests.Timeout:
        logger.error("Timeout al consultar Open Exchange Rates.")
        raise
    except requests.HTTPError as e:
        logger.error("Error HTTP en Open Exchange Rates: %s", e)
        raise

    data = response.json()

    try:
        exchange_info = {
            "base_currency": data["base_code"],
            "usd_rate":      data["rates"]["USD"],
            "last_updated":  data["time_last_update_utc"],
        }
    except KeyError as e:
        logger.error("Campo inesperado en respuesta de Exchange Rates: %s", e)
        raise

    logger.info(
        "Fuente 3 OK — 1 %s = %.6f USD (actualizado: %s)",
        exchange_info["base_currency"],
        exchange_info["usd_rate"],
        exchange_info["last_updated"],
    )
    return exchange_info
