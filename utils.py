# -*- coding: utf-8 -*-
"""
===============================================================================
utils.py — Funciones utilitarias compartidas (Adaptado para InfluxDB v3)
===============================================================================

Módulo con funciones reutilizables para todo el pipeline de detección
de anomalías con Autoencoder, adaptado para InfluxDB 3 usando Flight SQL.

===============================================================================
"""

import os
import sys
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

INFLUX_HOST = os.getenv("INFLUXDB_HOST", "192.168.10.145")
INFLUX_PORT = int(os.getenv("INFLUXDB_PORT", "8182"))
INFLUX_TOKEN = os.getenv("INFLUXDB_TOKEN", "apiv3_75b75fda71d81cda9bd2b417aaa5ea678b31467fd940d093")
INFLUX_DB = os.getenv("INFLUXDB_DATABASE", "sat_lab")
INFLUX_MEASUREMENT = os.getenv("INFLUXDB_MEASUREMENT", "mqtt_consumer")

MODELOS_DIR = os.path.join(BASE_DIR, "modelos")
DATOS_DIR = os.path.join(BASE_DIR, "datos")

# Definición de los nuevos sensores a monitorear
# Ahora todos forman parte de la consulta SQL como columnas/tags
SENSORES = {
    "PIT_001":               {"tipo": "REAL", "unidad": "PSI", "influx_col": "PIT_001/Pressure"},
    "FIT_001_MAS":           {"tipo": "REAL", "unidad": "kg/s", "influx_col": "FIT_001_MAS/Mass_Flow"},
    "FIT_001_DENS":          {"tipo": "REAL", "unidad": "kg/m3", "influx_col": "FIT_001_DENS/Density"},
    "FIT_001_TEMP":          {"tipo": "REAL", "unidad": "°C", "influx_col": "FIT_001_TEMP/Temperature"},
    "FIT_001_VOL":           {"tipo": "REAL", "unidad": "m3/s", "influx_col": "FIT_001_VOL/Volumetric_Flow"},
    "LIT_001":               {"tipo": "REAL", "unidad": "%", "influx_col": "LIT_001/Level"},
    "LIT_002":               {"tipo": "REAL", "unidad": "%", "influx_col": "LIT_002/Level"},
    "TT_001":                {"tipo": "REAL", "unidad": "°C", "influx_col": "TT_001/Temp"},
    "Bomba_Agua_001_STATUS": {"tipo": "BOOL", "unidad": "Estado", "influx_col": "Bomba_Agua_001_STATUS/State"},
    "Bomba_Agua_001_REF":    {"tipo": "REAL", "unidad": "RPM", "influx_col": "Bomba_Agua_001_REF/Speed_Ref"},
    "Val_001":               {"tipo": "BOOL", "unidad": "Estado", "influx_col": "Val_001/State"},
    "Val_002":               {"tipo": "BOOL", "unidad": "Estado", "influx_col": "Val_002/State"},
    "Val_003":               {"tipo": "BOOL", "unidad": "Estado", "influx_col": "Val_003/State"},
    "Val_004":               {"tipo": "BOOL", "unidad": "Estado", "influx_col": "Val_004/State"},
    "Mot_Comp_001":          {"tipo": "BOOL", "unidad": "Estado", "influx_col": "Mot_Comp_001/State"},
    "MOTOR_01":              {"tipo": "BOOL", "unidad": "Estado", "influx_col": "MOTOR_01/Running"},
}

COLUMNAS_FEATURES = list(SENSORES.keys())

def crear_directorios():
    os.makedirs(MODELOS_DIR, exist_ok=True)
    os.makedirs(DATOS_DIR, exist_ok=True)
    print(f"[OK] Directorios verificados:")
    print(f"     - Modelos: {MODELOS_DIR}")
    print(f"     - Datos:   {DATOS_DIR}")


# =============================================================================
# CONSULTA A INFLUXDB v3 (Flight SQL)
# =============================================================================

from influxdb_client_3 import InfluxDBClient3

def consultar_sensor_influxdb(tag_name, inicio, fin):
    """
    Consulta una serie temporal (un tag) usando InfluxDB 3 (Flight SQL) via influxdb_client_3.
    """
    print(f"  -> Consultando {tag_name} desde InfluxDB (FlightSQL)...")
    
    # IMPORTANTE: InfluxDB v3 SQL usa sintaxis PostgreSQL
    # Convirtiendo fechas a strings compatibles (timestamp 'YYYY-MM-DD HH:MM:SS')
    inicio_str = inicio.strftime('%Y-%m-%d %H:%M:%S')
    fin_str = fin.strftime('%Y-%m-%d %H:%M:%S')
    
    # Asumimos que la métrica o el nombre de la variable es una columna en el measurement
    # Si devuelve error de columna inexistente, ajustar a la segunda forma comentada arriba.
    influx_col = SENSORES.get(tag_name, {}).get("influx_col", tag_name)
    query = f"""
    SELECT 
        time as "timestamp", 
        "{influx_col}" as "{tag_name}"
    FROM "{INFLUX_MEASUREMENT}"
    WHERE time >= timestamp '{inicio_str}'
      AND time <= timestamp '{fin_str}'
      AND "{influx_col}" IS NOT NULL
    ORDER BY time ASC
    """
    
    try:
        client = InfluxDBClient3(host=f"grpc://{INFLUX_HOST}:{INFLUX_PORT}", token=INFLUX_TOKEN, database=INFLUX_DB)
        table = client.query(query=query, language="sql")
        df = table.to_pandas()
        
        if df.empty:
            print(f"    [WARN] No se encontraron datos para {tag_name}")
            return pd.DataFrame()
            
        # Asegurar formato correcto de timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        # Limpiar duplicados si los hay y ordenar
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        
        print(f"    [OK] {len(df):,} muestras obtenidas. Rango: {df['timestamp'].min()} -> {df['timestamp'].max()}")
        return df
    except Exception as e:
        print(f"    [ERROR] Falló la consulta a InfluxDB para {tag_name}: {e}")
        # Hint para el usuario si es por esquema
        if "column" in str(e).lower() and "does not exist" in str(e).lower():
            print(f"    [!] Asegúrate de que '{INFLUX_MEASUREMENT}' es la tabla correcta y '{tag_name}' es una columna.")
        return pd.DataFrame()


# =============================================================================
# FUNCIONES DE CARGA
# =============================================================================

def cargar_datos_csv(nombre_archivo="datos_sensores.csv"):
    ruta = os.path.join(DATOS_DIR, nombre_archivo)
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró: {ruta}")
    df = pd.read_csv(ruta, parse_dates=["timestamp"], index_col="timestamp")
    return df

def cargar_modelo(nombre_modelo="autoencoder_anomalias.keras"):
    from tensorflow import keras
    ruta = os.path.join(MODELOS_DIR, nombre_modelo)
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró: {ruta}")
    return keras.models.load_model(ruta)

def cargar_scaler(nombre_archivo="scaler.joblib"):
    ruta = os.path.join(MODELOS_DIR, nombre_archivo)
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró: {ruta}")
    return joblib.load(ruta)

def cargar_umbral(nombre_archivo="umbral.joblib"):
    ruta = os.path.join(MODELOS_DIR, nombre_archivo)
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró: {ruta}")
    return joblib.load(ruta)

if __name__ == "__main__":
    print("Utilidades InfluxDB v3 FlightSQL")
    crear_directorios()
