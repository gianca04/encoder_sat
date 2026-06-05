# -*- coding: utf-8 -*-
"""
===============================================================================
utils.py — Funciones utilitarias compartidas
===============================================================================

Módulo con funciones reutilizables para todo el pipeline de detección
de anomalías con Autoencoder.

Funciones principales:
    - consultar_prometheus(): Wrapper para API query_range de Prometheus/VictoriaMetrics
    - cargar_datos_csv():     Carga el dataset alineado desde CSV
    - cargar_modelo():        Carga un modelo Keras guardado (.keras)
    - cargar_scaler():        Carga el scaler de normalización (pickle/joblib)

Autor: Autoencoder Anomaly Detection Pipeline
Fecha: 2026-06-01
===============================================================================
"""

import os
import sys
import requests
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Forzar UTF-8 en la consola de Windows (evita errores cp1252)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


# =============================================================================
# CONFIGURACIÓN GLOBAL
# =============================================================================

# Directorio base del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Cargar variables de entorno
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Endpoint de VictoriaMetrics (compatible con API Prometheus)
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://192.168.10.58:8428/prometheus")

# Directorio para guardar modelos y artefactos
MODELOS_DIR = os.path.join(BASE_DIR, "modelos")
DATOS_DIR = os.path.join(BASE_DIR, "datos")

# Definición de los sensores a monitorear
# Formato: (equipo, metrica, tipo_dato, valor_nominal, variacion_normal)
SENSORES = {
    "FIT_001_Flow":     {"equipo": "FIT_001",  "metrica": "Flow",    "tipo": "REAL", "unidad": "m³/h", "nominal": 7.5,  "variacion": 0.5},
    "LIT_001_Level":    {"equipo": "LIT_001",  "metrica": "Level",   "tipo": "REAL", "unidad": "%",    "nominal": 15.0, "variacion": 2.0},
    "LIT_002_Level":    {"equipo": "LIT_002",  "metrica": "Level",   "tipo": "REAL", "unidad": "%",    "nominal": 15.0, "variacion": 2.0},
    "TT_001_Temp":      {"equipo": "TT_001",   "metrica": "Temp",    "tipo": "REAL", "unidad": "°C",   "nominal": 30.0, "variacion": 1.0},
    "TT_002_Temp":      {"equipo": "TT_002",   "metrica": "Temp",    "tipo": "REAL", "unidad": "°C",   "nominal": 30.0, "variacion": 1.0},
}

# Nombres de columnas en orden (para el modelo)
COLUMNAS_FEATURES = list(SENSORES.keys())


def crear_directorios():
    """
    Crea los directorios necesarios para el proyecto si no existen.
    
    Directorios creados:
        - modelos/  -> Para guardar el modelo .keras, scaler, y umbral
        - datos/    -> Para guardar CSVs intermedios
    """
    os.makedirs(MODELOS_DIR, exist_ok=True)
    os.makedirs(DATOS_DIR, exist_ok=True)
    print(f"[OK] Directorios verificados:")
    print(f"     - Modelos: {MODELOS_DIR}")
    print(f"     - Datos:   {DATOS_DIR}")


# =============================================================================
# FUNCIONES DE CONSULTA A PROMETHEUS / VICTORIAMETRICS
# =============================================================================

def consultar_prometheus(equipo, metrica, inicio, fin, step="15s"):
    """
    Consulta una serie temporal desde la API query_range de Prometheus/VictoriaMetrics.
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │  EXPLICACIÓN DEL PROCESO:                                         │
    │                                                                   │
    │  La API query_range permite obtener valores de una métrica en un  │
    │  rango de tiempo definido. Cada consulta retorna pares            │
    │  [timestamp, valor] que luego se convierten a DataFrame.          │
    │                                                                   │
    │  VictoriaMetrics es compatible con la API de Prometheus, por lo   │
    │  que usamos el endpoint /api/v1/query_range estándar.             │
    └─────────────────────────────────────────────────────────────────────┘
    
    Parámetros:
    -----------
    equipo : str
        Nombre del equipo (ej: "FIT_001")
    metrica : str
        Nombre de la métrica (ej: "Flow")
    inicio : datetime
        Timestamp de inicio de la consulta
    fin : datetime
        Timestamp de fin de la consulta
    step : str
        Intervalo de muestreo (default: "15s")
        
    Retorna:
    --------
    pd.DataFrame
        DataFrame con columnas ['timestamp', '<equipo>_<metrica>']
        El timestamp está en formato datetime UTC.
    """
    # Construir la query PromQL
    query = f'lab_sat_valor{{equipo="{equipo}", metrica="{metrica}"}}'
    
    # Parámetros para la API query_range
    params = {
        "query": query,
        "start": int(inicio.timestamp()),
        "end":   int(fin.timestamp()),
        "step":  step,
    }
    
    url = f"{PROMETHEUS_URL}/api/v1/query_range"
    
    print(f"  -> Consultando: {equipo}/{metrica} ...")
    print(f"    URL: {url}")
    print(f"    Query: {query}")
    print(f"    Rango: {inicio} -> {fin} (step={step})")
    
    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        # Validar respuesta
        if data.get("status") != "success":
            print(f"    [ERROR] Status: {data.get('status')}")
            print(f"    Error: {data.get('error', 'Desconocido')}")
            return pd.DataFrame()
        
        # Extraer los valores de la serie temporal
        results = data.get("data", {}).get("result", [])
        
        if not results:
            print(f"    [WARN] No se encontraron datos para {equipo}/{metrica}")
            return pd.DataFrame()
        
        # Tomar la primera serie (debería ser la única)
        values = results[0].get("values", [])
        nombre_columna = f"{equipo}_{metrica}"
        
        # Convertir a DataFrame
        df = pd.DataFrame(values, columns=["timestamp", nombre_columna])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df[nombre_columna] = df[nombre_columna].astype(float)
        
        print(f"    [OK] {len(df)} muestras obtenidas")
        print(f"    Rango real: {df['timestamp'].min()} -> {df['timestamp'].max()}")
        
        return df
        
    except requests.exceptions.ConnectionError:
        print(f"    [ERROR] No se pudo conectar a {PROMETHEUS_URL}")
        print(f"    Verifica que VictoriaMetrics esté corriendo en esa dirección.")
        return pd.DataFrame()
    except requests.exceptions.Timeout:
        print(f"    [ERROR] Timeout al consultar {equipo}/{metrica}")
        return pd.DataFrame()
    except Exception as e:
        print(f"    [ERROR] Error inesperado: {type(e).__name__}: {e}")
        return pd.DataFrame()


# =============================================================================
# FUNCIONES DE CARGA DE ARCHIVOS
# =============================================================================

def cargar_datos_csv(nombre_archivo="datos_sensores.csv"):
    """
    Carga el dataset de sensores desde un archivo CSV.
    
    El CSV debe tener una columna 'timestamp' y las columnas de features
    definidas en COLUMNAS_FEATURES.
    
    Parámetros:
    -----------
    nombre_archivo : str
        Nombre del archivo CSV dentro del directorio datos/
        
    Retorna:
    --------
    pd.DataFrame
        DataFrame con timestamp como índice y las features como columnas.
    """
    ruta = os.path.join(DATOS_DIR, nombre_archivo)
    
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontró el archivo: {ruta}\n"
            f"Ejecuta primero '01_extraer_datos.py' para generar los datos."
        )
    
    df = pd.read_csv(ruta, parse_dates=["timestamp"], index_col="timestamp")
    print(f"[OK] Datos cargados desde: {ruta}")
    print(f"     Shape: {df.shape}")
    print(f"     Rango: {df.index.min()} -> {df.index.max()}")
    print(f"     Columnas: {list(df.columns)}")
    
    return df


def cargar_modelo(nombre_modelo="autoencoder_anomalias.keras"):
    """
    Carga un modelo Keras guardado en formato .keras
    
    Parámetros:
    -----------
    nombre_modelo : str
        Nombre del archivo del modelo dentro del directorio modelos/
        
    Retorna:
    --------
    tensorflow.keras.Model
        Modelo Keras cargado y listo para inferencia.
    """
    from tensorflow import keras
    
    ruta = os.path.join(MODELOS_DIR, nombre_modelo)
    
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontró el modelo: {ruta}\n"
            f"Ejecuta primero '03_entrenar_autoencoder.py' para entrenar el modelo."
        )
    
    modelo = keras.models.load_model(ruta)
    print(f"[OK] Modelo cargado desde: {ruta}")
    modelo.summary()
    
    return modelo


def cargar_scaler(nombre_archivo="scaler.joblib"):
    """
    Carga el scaler de normalización guardado con joblib.
    
    Parámetros:
    -----------
    nombre_archivo : str
        Nombre del archivo del scaler dentro del directorio modelos/
        
    Retorna:
    --------
    sklearn.preprocessing.MinMaxScaler
        Scaler ajustado, listo para transformar nuevos datos.
    """
    ruta = os.path.join(MODELOS_DIR, nombre_archivo)
    
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontró el scaler: {ruta}\n"
            f"Ejecuta primero '02_preprocesar_datos.py' para generar el scaler."
        )
    
    scaler = joblib.load(ruta)
    print(f"[OK] Scaler cargado desde: {ruta}")
    
    return scaler


def cargar_umbral(nombre_archivo="umbral.joblib"):
    """
    Carga el umbral de anomalía guardado con joblib.
    
    Parámetros:
    -----------
    nombre_archivo : str
        Nombre del archivo del umbral dentro del directorio modelos/
        
    Retorna:
    --------
    dict
        Diccionario con claves 'p95', 'p99', 'mean', 'std'.
    """
    ruta = os.path.join(MODELOS_DIR, nombre_archivo)
    
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontró el umbral: {ruta}\n"
            f"Ejecuta primero '03_entrenar_autoencoder.py' para calcular el umbral."
        )
    
    umbral = joblib.load(ruta)
    print(f"[OK] Umbral cargado desde: {ruta}")
    print(f"     Percentil 95: {umbral['p95']:.6f}")
    print(f"     Percentil 99: {umbral['p99']:.6f}")
    print(f"     Media:        {umbral['mean']:.6f}")
    print(f"     Std:          {umbral['std']:.6f}")
    
    return umbral


# =============================================================================
# UTILIDADES DE VISUALIZACIÓN
# =============================================================================

def estilo_grafica():
    """
    Aplica un estilo consistente a las gráficas de matplotlib.
    Debe llamarse antes de crear figuras.
    """
    import matplotlib.pyplot as plt
    
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams.update({
        "figure.figsize": (14, 8),
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "lines.linewidth": 1.0,
        "figure.dpi": 100,
    })


if __name__ == "__main__":
    print("=" * 60)
    print("  UTILIDADES — Autoencoder Detector de Anomalías")
    print("=" * 60)
    print()
    print(f"Prometheus URL: {PROMETHEUS_URL}")
    print(f"Base Dir:       {BASE_DIR}")
    print(f"Modelos Dir:    {MODELOS_DIR}")
    print(f"Datos Dir:      {DATOS_DIR}")
    print()
    print("Sensores configurados:")
    for nombre, config in SENSORES.items():
        print(f"  {nombre:25s} -> tipo={config['tipo']}, "
              f"nominal={config['nominal']}, var={config['variacion']}")
    print()
    crear_directorios()
