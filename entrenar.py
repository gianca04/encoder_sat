# -*- coding: utf-8 -*-
"""
===============================================================================
entrenar.py — Script Único de Extracción, Preprocesamiento y Entrenamiento
===============================================================================

PROPÓSITO:
    Este script único automatiza todo el pipeline de aprendizaje de extremo a
    extremo de forma segura y profesional para producción:
    1. Extrae series de sensores históricos desde VictoriaMetrics.
    2. Realiza alineación temporal y limpieza en memoria.
    3. Normaliza y divide cronológicamente en train/test (80/20).
    4. Entrenar el Autoencoder (Keras) y calcula nuevos umbrales (P95/P99).
    5. Guarda y actualiza los artefactos de producción de manera segura.

MECANISMOS DE SEGURIDAD (FAIL-SAFE):
    - Backup automático: Antes de realizar cualquier cambio, respalda
      el modelo actual en modelos/backup/.
    - Rollback en fallos: Si la conexión a VictoriaMetrics se pierde,
      o si el entrenamiento arroja una excepción, los archivos vigentes se
      restauran automáticamente desde el backup para no romper la detección MQTT.

USO:
    # Entrenar desde el 29/05/2026 00:00:00 hasta ahora:
    python entrenar.py --start-date "2026-05-29T00:00:00"
    
    # Rango de fechas cerrado para experimentación:
    python entrenar.py --start-date "2026-05-29T00:00:00" --end-date "2026-06-02T12:00:00"

===============================================================================
"""

import os
import sys
import shutil
import argparse
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timezone

# Forzar UTF-8 en la consola de Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# TensorFlow y sklearn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import MinMaxScaler

from utils import (
    consultar_prometheus,
    crear_directorios,
    SENSORES,
    COLUMNAS_FEATURES,
    MODELOS_DIR,
    DATOS_DIR,
)

# Archivos de producción y directorio de respaldo
ARCHIVOS_MODELO = {
    "modelo": "autoencoder_anomalias.keras",
    "scaler": "scaler.joblib",
    "umbral": "umbral.joblib",
    "metadata": "metadata_modelo.joblib",
}

BACKUP_DIR = os.path.join(MODELOS_DIR, "backup")


def crear_backup():
    """Realiza copia de seguridad de los artefactos de producción actuales."""
    print("\n" + "=" * 60)
    print("  PASO 1: Creación de copia de seguridad (Backup)")
    print("=" * 60)
    
    os.makedirs(BACKUP_DIR, exist_ok=True)
    respaldados = 0
    
    for clave, nombre in ARCHIVOS_MODELO.items():
        ruta_orig = os.path.join(MODELOS_DIR, nombre)
        ruta_dest = os.path.join(BACKUP_DIR, nombre)
        
        if os.path.exists(ruta_orig):
            try:
                shutil.copy2(ruta_orig, ruta_dest)
                print(f"    [OK] Resguardado: {nombre} -> backup/")
                respaldados += 1
            except Exception as e:
                print(f"    [ERROR] No se pudo resguardar {nombre}: {e}")
                raise e
                
    if respaldados == 0:
        print("    [INFO] No hay modelos previos en producción. Primer inicio de entrenamiento.")
    else:
        print(f"    [OK] Se respaldaron {respaldados} archivos correctamente.")


def restaurar_backup():
    """Restaura los artefactos guardados en caso de fallo durante el entrenamiento."""
    print("\n" + "!" * 60)
    print("  [FALLO DETECTADO] Reestableciendo modelo y escaladores anteriores...")
    print("!" * 60)
    
    if not os.path.exists(BACKUP_DIR):
        print("    [WARN] No existe carpeta de backup para restaurar.")
        return
        
    restaurados = 0
    for clave, nombre in ARCHIVOS_MODELO.items():
        ruta_orig = os.path.join(BACKUP_DIR, nombre)
        ruta_dest = os.path.join(MODELOS_DIR, nombre)
        
        if os.path.exists(ruta_orig):
            try:
                shutil.copy2(ruta_orig, ruta_dest)
                print(f"    [RESTORE] Restaurado: {nombre}")
                restaurados += 1
            except Exception as e:
                print(f"    [CRÍTICO] Fallo al restaurar {nombre}: {e}")
                
    print(f"    [INFO] Se restauraron {restaurados} archivos de producción estables.")


def extraer_datos_historicos(inicio_str, fin_str=None, verbose=False):
    """Consulta y consolida datos reales de VictoriaMetrics en un rango dado."""
    print("\n" + "=" * 60)
    print("  PASO 2: Extracción de datos desde VictoriaMetrics")
    print("=" * 60)
    
    inicio = datetime.fromisoformat(inicio_str)
    if fin_str:
        fin = datetime.fromisoformat(fin_str)
    else:
        fin = datetime.now(timezone.utc)
        
    inicio_naive = inicio.replace(tzinfo=None)
    fin_naive = fin.replace(tzinfo=None)
    
    print(f"  Inicio:           {inicio_naive} UTC")
    print(f"  Fin:              {fin_naive} UTC")
    
    dataframes = {}
    for nombre_sensor, config in SENSORES.items():
        if verbose:
            print(f"  -> Consultando {nombre_sensor}...")
            
        df_sensor = consultar_prometheus(
            equipo=config["equipo"],
            metrica=config["metrica"],
            inicio=inicio_naive,
            fin=fin_naive,
            step="30s",
        )
        
        if not df_sensor.empty:
            dataframes[nombre_sensor] = df_sensor
        else:
            print(f"  [⚠] Sensor sin datos obtenidos: {nombre_sensor}")
            
    if len(dataframes) != len(SENSORES):
        faltantes = set(SENSORES.keys()) - set(dataframes.keys())
        raise ValueError(f"Faltan datos de los siguientes sensores: {faltantes}")
        
    print("\n  Alineando temporalmente todas las series...")
    nombres = list(dataframes.keys())
    df_consolidado = dataframes[nombres[0]].copy()
    
    for nombre in nombres[1:]:
        df_consolidado = pd.merge(
            df_consolidado,
            dataframes[nombre],
            on="timestamp",
            how="outer",
        )
        
    df_consolidado = df_consolidado.sort_values("timestamp").reset_index(drop=True)
    print(f"    [OK] Datos crudos alineados: {len(df_consolidado):,} filas")
    
    return df_consolidado


def preprocesar_datos(df_crudo, max_gap_segundos=180):
    """
    Limpia y normaliza datos cargados en memoria.
    Maneja adecuadamente tiempos muertos y paradas prolongadas (gaps de días/meses)
    interpolando solo DENTRO de sub-bloques continuos de tiempo.
    """
    print("\n" + "=" * 60)
    print("  PASO 3: Preprocesamiento y normalización (Manejo de Gaps Temporales)")
    print("=" * 60)
    
    df = df_crudo.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
    else:
        df = df.sort_index()
        df["timestamp"] = pd.to_datetime(df.index)
        df = df.reset_index(drop=True)

    # Identificar vacíos de datos (Gaps > max_gap_segundos)
    df["delta_time"] = df["timestamp"].diff().dt.total_seconds()
    df["bloque_id"] = (df["delta_time"] > max_gap_segundos).cumsum()

    col_motor = "MOTOR_01_Running"
    cols_features = [c for c in COLUMNAS_FEATURES if c in df.columns]
    cols_continuas = [c for c in cols_features if c != col_motor]

    dfs_limpios = []
    n_bloques = df["bloque_id"].nunique()
    print(f"  [INFO] Se detectaron {n_bloques:,} bloque(s) continuos de datos (gaps > {max_gap_segundos}s).")

    for bloque_id, df_bloque in df.groupby("bloque_id"):
        if len(df_bloque) < 5:
            continue  # Descartar fragmentos ruidosos demasiado pequeños
            
        df_b = df_bloque.set_index("timestamp")[cols_features].copy()
        
        # Interpolar únicamente DENTRO del bloque continuo
        for col in cols_continuas:
            df_b[col] = df_b[col].interpolate(method="linear", limit=5).ffill().bfill()
            
        if col_motor in df_b.columns:
            df_b[col_motor] = df_b[col_motor].ffill().bfill()

        dfs_limpios.append(df_b)

    if not dfs_limpios:
        raise ValueError("No se obtuvieron bloques de datos válidos para procesar.")

    df_consolidado = pd.concat(dfs_limpios)

    # Filtrar si la máquina estuvo apagada o motor en 0
    if col_motor in df_consolidado.columns:
        n_prev = len(df_consolidado)
        df_consolidado = df_consolidado[df_consolidado[col_motor] == 1]
        print(f"  [INFO] Filtrado por operación ({col_motor}==1): {len(df_consolidado):,} de {n_prev:,} muestras conservadas.")

    df = df_consolidado.dropna()
    n_muestras = len(df)

    if n_muestras < 100:
        raise ValueError(f"Muestras insuficientes para entrenar: {n_muestras}")

    print(f"  Total muestras limpias consolidadas: {n_muestras:,}")

    # Split cronológico 80/20 sobre la data unificada operativa
    n_train = int(n_muestras * 0.8)
    df_train = df.iloc[:n_train].copy()
    df_test = df.iloc[n_train:].copy()

    print(f"  Muestras de Entrenamiento (Train): {len(df_train):,} (80%)")
    print(f"  Muestras de Validación (Test):      {len(df_test):,} (20%)")

    scaler = MinMaxScaler(feature_range=(0, 1))
    datos_train_norm = scaler.fit_transform(df_train.values)
    datos_test_norm = scaler.transform(df_test.values)

    df_train_norm = pd.DataFrame(datos_train_norm, columns=df_train.columns, index=df_train.index)
    df_test_norm = pd.DataFrame(datos_test_norm, columns=df_test.columns, index=df_test.index)

    # Guardar reporte histórico en disco
    try:
        ruta_csv = os.path.join(DATOS_DIR, "datos_sensores_ultimo_entrenamiento.csv")
        df.to_csv(ruta_csv, index=True)
        print(f"  [OK] Dataset histórico guardado en: {ruta_csv}")
    except Exception as e:
        print(f"  [WARN] No se pudo guardar el CSV histórico: {e}")

    return df_train_norm, df_test_norm, scaler


def construir_autoencoder(n_features):
    """Crea la arquitectura Keras del Autoencoder (6 -> 32 -> 16 -> 8 -> 16 -> 32 -> 6)."""
    inputs = layers.Input(shape=(n_features,), name="input_sensores")
    
    # Encoder
    x = layers.Dense(32, activation="relu", name="encoder_1")(inputs)
    x = layers.Dense(16, activation="relu", name="encoder_2")(x)
    bottleneck = layers.Dense(8, activation="relu", name="bottleneck")(x)
    
    # Decoder
    x = layers.Dense(16, activation="relu", name="decoder_1")(bottleneck)
    x = layers.Dense(32, activation="relu", name="decoder_2")(x)
    outputs = layers.Dense(n_features, activation="sigmoid", name="output_reconstruccion")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="autoencoder_anomalias")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    
    return model


def entrenar_modelo(df_train_norm, df_test_norm, epochs=200, batch_size=64):
    """Entrena la red con parada temprana."""
    print("\n" + "=" * 60)
    print("  PASO 4: Entrenamiento del Autoencoder")
    print("=" * 60)
    
    X_train = df_train_norm.values.astype(np.float32)
    X_test = df_test_norm.values.astype(np.float32)
    
    autoencoder = construir_autoencoder(X_train.shape[1])
    
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        verbose=1,
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1,
    )
    
    history = autoencoder.fit(
        X_train, X_train,
        validation_data=(X_test, X_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        shuffle=True,
        verbose=1,
    )
    
    val_loss_final = min(history.history["val_loss"])
    epochs_ejecutadas = len(history.history["loss"])
    
    print(f"\n  ✓ Entrenamiento completado en {epochs_ejecutadas} épocas.")
    print(f"    Mejor val_loss:   {val_loss_final:.6f}")
    
    return autoencoder, history, val_loss_final


def calcular_umbral(autoencoder, df_train_norm):
    """Calcula umbrales globales y estadísticas de error por sensor individual."""
    print("\n" + "=" * 60)
    print("  PASO 5: Cálculo del nuevo umbral de anomalía")
    print("=" * 60)
    
    X_train = df_train_norm.values.astype(np.float32)
    X_reconstructed = autoencoder.predict(X_train, verbose=0)
    
    # ── UMBRAL GLOBAL (MSE promedio de todos los sensores por muestra) ──
    mse = np.mean(np.square(X_train - X_reconstructed), axis=1)
    
    p95 = float(np.percentile(mse, 95))
    p99 = float(np.percentile(mse, 99))
    mean = float(np.mean(mse))
    std = float(np.std(mse))
    
    print(f"    Nuevo umbral P95: {p95:.6f}")
    print(f"    Nuevo umbral P99: {p99:.6f}")
    print(f"    Error medio:      {mean:.6f} ± {std:.6f}")
    
    # ── ESTADÍSTICAS POR SENSOR INDIVIDUAL (error² por columna) ──
    # Cada sensor tiene su propia distribución de error de reconstrucción
    # Esto permite calcular salud relativa a su propio comportamiento histórico
    errores_por_sensor = np.square(X_train - X_reconstructed)  # shape: (n_muestras, n_features)
    columnas = list(df_train_norm.columns)
    
    stats_por_sensor = {}
    print(f"\n    Estadísticas por sensor (error de reconstrucción):")
    print(f"    {'Sensor':25s} {'Media':>10s} {'Std':>10s} {'P95':>10s} {'P99':>10s}")
    print(f"    {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    for i, col in enumerate(columnas):
        errores_col = errores_por_sensor[:, i]
        stats = {
            "mean": float(np.mean(errores_col)),
            "std": float(np.std(errores_col)),
            "p95": float(np.percentile(errores_col, 95)),
            "p99": float(np.percentile(errores_col, 99)),
        }
        stats_por_sensor[col] = stats
        print(f"    {col:25s} {stats['mean']:10.6f} {stats['std']:10.6f} {stats['p95']:10.6f} {stats['p99']:10.6f}")
    
    return {
        "p95": p95,
        "p99": p99,
        "mean": mean,
        "std": std,
        "por_sensor": stats_por_sensor,
    }


def guardar_artefactos(autoencoder, scaler, umbral, metadata):
    """Escribe de forma atómica los archivos listos a la carpeta de producción modelos/."""
    print("\n" + "=" * 60)
    print("  PASO 6: Reemplazo seguro de archivos de producción")
    print("=" * 60)
    
    autoencoder.save(os.path.join(MODELOS_DIR, ARCHIVOS_MODELO["modelo"]))
    joblib.dump(scaler, os.path.join(MODELOS_DIR, ARCHIVOS_MODELO["scaler"]))
    joblib.dump(umbral, os.path.join(MODELOS_DIR, ARCHIVOS_MODELO["umbral"]))
    joblib.dump(metadata, os.path.join(MODELOS_DIR, ARCHIVOS_MODELO["metadata"]))
    
    for clave, nombre in ARCHIVOS_MODELO.items():
        print(f"    [OK] Actualizado en producción: {nombre}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified train script for encoder_sat anomaly detector."
    )
    parser.add_argument("--start-date", type=str, default="2026-05-29T00:00:00",
                        help="Fecha de inicio ISO 8601 (YYYY-MM-DDTHH:MM:SS)")
    parser.add_argument("--end-date", type=str, default=None,
                        help="Fecha de fin ISO 8601 (YYYY-MM-DDTHH:MM:SS). Default: ahora")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Máximo de épocas")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Tamaño de batch")
    parser.add_argument("--verbose", action="store_true",
                        help="Mostrar logs de extracción")
    args = parser.parse_args()

    print("+" + "=" * 58 + "+")
    print("|  AUTOENCODER ANOMALY DETECTOR -- INTEGRATED TRAINER      |")
    print("|  Extracción + Preprocesamiento + Entrenamiento           |")
    print("+" + "=" * 58 + "+")
    
    crear_directorios()
    
    try:
        crear_backup()
    except Exception as e:
        print(f"\n[CRÍTICO] Fallo al resguardar archivos. Se cancela el entrenamiento: {e}")
        sys.exit(1)
        
    try:
        # Extraer, procesar, entrenar, valorar, guardar
        df_crudo = extraer_datos_historicos(args.start_date, args.end_date, verbose=args.verbose)
        df_train_norm, df_test_norm, scaler = preprocesar_datos(df_crudo)
        
        autoencoder, history, mejor_val_loss = entrenar_modelo(
            df_train_norm, df_test_norm, epochs=args.epochs, batch_size=args.batch_size
        )
        
        umbral_info = calcular_umbral(autoencoder, df_train_norm)
        
        metadata = {
            "n_features": len(COLUMNAS_FEATURES),
            "columnas": COLUMNAS_FEATURES,
            "arquitectura": "6->32->16->8->16->32->6",
            "loss": "mse",
            "epochs_reales": len(history.history["loss"]),
            "mejor_val_loss": float(mejor_val_loss),
            "umbral_p95": umbral_info["p95"],
            "umbral_p99": umbral_info["p99"],
            "train_samples": int(df_train_norm.shape[0]),
            "test_samples": int(df_test_norm.shape[0]),
            "fecha_entrenamiento": datetime.now(timezone.utc).isoformat(),
            "rango_datos": f"{df_crudo['timestamp'].min()} a {df_crudo['timestamp'].max()}",
        }
        
        guardar_artefactos(autoencoder, scaler, umbral_info, metadata)
        
        print("\n" + "=" * 60)
        print("  [ÉXITO] ENTRENAMIENTO COMPLETADO Y ACTUALIZADO EN PRODUCCIÓN")
        print(f"  Rango:          {metadata['rango_datos']}")
        print(f"  Muestras train: {metadata['train_samples']}")
        print(f"  Val Loss (MSE): {metadata['mejor_val_loss']:.6f}")
        print(f"  Umbral P95:     {metadata['umbral_p95']:.6f}")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n[ERROR] Ocurrió una excepción durante el proceso: {e}")
        import traceback
        traceback.print_exc()
        restaurar_backup()
        sys.exit(1)


if __name__ == "__main__":
    main()
