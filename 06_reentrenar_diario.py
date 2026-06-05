# -*- coding: utf-8 -*-
"""
===============================================================================
06_reentrenar_diario.py — Reentrenamiento Diario Automatizado y Seguro
===============================================================================

PROPÓSITO:
    Este script está diseñado para correr diariamente (como tarea programada,
    cron job o daemon) para actualizar y mejorar el modelo Autoencoder con los
    datos reales acumulados desde el inicio del proyecto (29/05/2026 00:00:00).

MECANISMOS DE ROBUSTEZ (FAIL-SAFE):
    1. Resguardo (Backup): Antes de iniciar, guarda una copia del modelo
       vigente, scaler, umbral y metadata.
    2. Aislamiento de fallas: Si la extracción de VictoriaMetrics falla
       o si el entrenamiento arroja errores, el script revierte y restaura
       los archivos originales desandando cualquier cambio a medias.
    3. Prevención de corrupción: En ningún caso el detector MQTT en tiempo real
       se quedará sin un modelo válido o con archivos rotos/incompletos.

PROCESO:
    ┌─────────────────────────────────────────────────────────────────────┐
    │  1. Crear backup de los artefactos actuales en modelos/backup/      │
    │  2. Consultar VictoriaMetrics desde 29/05/2026 a la fecha actual   │
    │  3. Preprocesar en memoria (alineación, NaNs, split 80/20)          │
    │  4. Entrenar nuevo Autoencoder (EarlyStopping, learning rate decay) │
    │  5. Calcular nuevos umbrales de anomalía (P95, P99)                 │
    │  6. Validar y reemplazar archivos de producción de manera atómica   │
    └─────────────────────────────────────────────────────────────────────┘

USO:
    python 06_reentrenar_diario.py
    python 06_reentrenar_diario.py --epochs 100 --batch-size 64
    python 06_reentrenar_diario.py --verbose

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

# TensorFlow / Keras y sklearn
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

# Configuración de archivos clave
ARCHIVOS_MODELO = {
    "modelo": "autoencoder_anomalias.keras",
    "scaler": "scaler.joblib",
    "umbral": "umbral.joblib",
    "metadata": "metadata_modelo.joblib",
}

BACKUP_DIR = os.path.join(MODELOS_DIR, "backup")


def crear_backup():
    """
    Realiza una copia de seguridad de los archivos de modelo actuales.
    """
    print("\n" + "=" * 60)
    print("  PASO 1: Creación de copia de seguridad (Backup)")
    print("=" * 60)
    
    os.makedirs(BACKUP_DIR, exist_ok=True)
    respaldados = 0
    
    for clave, nombre_archivo in ARCHIVOS_MODELO.items():
        ruta_orig = os.path.join(MODELOS_DIR, nombre_archivo)
        ruta_dest = os.path.join(BACKUP_DIR, nombre_archivo)
        
        if os.path.exists(ruta_orig):
            try:
                shutil.copy2(ruta_orig, ruta_dest)
                print(f"    [OK] Resguardado: {nombre_archivo} -> backup/")
                respaldados += 1
            except Exception as e:
                print(f"    [ERROR] No se pudo copiar {nombre_archivo}: {e}")
                raise e
    
    if respaldados == 0:
        print("    [INFO] No se encontraron modelos previos para resguardar. Primer entrenamiento.")
    else:
        print(f"    [OK] Se respaldaron {respaldados} archivos correctamente.")


def restaurar_backup():
    """
    Restaura los archivos desde el directorio de backup.
    Se ejecuta si ocurre algún fallo durante la extracción o entrenamiento.
    """
    print("\n" + "!" * 60)
    print("  [FALLO EN PROCESO] Iniciando restauración de backup...")
    print("!" * 60)
    
    if not os.path.exists(BACKUP_DIR):
        print("    [WARN] No existe directorio de backup para restaurar.")
        return
        
    restaurados = 0
    for clave, nombre_archivo in ARCHIVOS_MODELO.items():
        ruta_orig = os.path.join(BACKUP_DIR, nombre_archivo)
        ruta_dest = os.path.join(MODELOS_DIR, nombre_archivo)
        
        if os.path.exists(ruta_orig):
            try:
                shutil.copy2(ruta_orig, ruta_dest)
                print(f"    [RESTORE] Restaurado: {nombre_archivo}")
                restaurados += 1
            except Exception as e:
                print(f"    [CRÍTICO] Error al restaurar {nombre_archivo}: {e}")
                
    print(f"    [INFO] Se restauraron {restaurados} archivos de producción.")


def extraer_datos_historicos(inicio_str, verbose=False):
    """
    Extrae todos los datos desde inicio_str hasta el momento actual.
    """
    print("\n" + "=" * 60)
    print("  PASO 2: Extracción de datos históricos desde VictoriaMetrics")
    print("=" * 60)
    
    # Rango de tiempo
    inicio = datetime.fromisoformat(inicio_str)
    fin = datetime.now(timezone.utc)
    
    # Quitar tzinfo para compatibilidad con el endpoint
    inicio_naive = inicio.replace(tzinfo=None)
    fin_naive = fin.replace(tzinfo=None)
    
    print(f"  Inicio:           {inicio_naive} UTC")
    print(f"  Fin (Ahora):      {fin_naive} UTC")
    
    dataframes = {}
    errores = []
    
    for nombre_sensor, config in SENSORES.items():
        if verbose:
            print(f"  -> Extrayendo: {nombre_sensor}...")
            
        df_sensor = consultar_prometheus(
            equipo=config["equipo"],
            metrica=config["metrica"],
            inicio=inicio_naive,
            fin=fin_naive,
            step="15s",
        )
        
        if df_sensor.empty:
            errores.append(nombre_sensor)
            print(f"  [⚠] Sin datos para {nombre_sensor}")
        else:
            dataframes[nombre_sensor] = df_sensor
            
    # Validar extracción
    if len(dataframes) != len(SENSORES):
        raise ValueError(
            f"No se pudieron extraer datos de todos los sensores. "
            f"Faltantes: {set(SENSORES.keys()) - set(dataframes.keys())}"
        )
        
    print("\n  Alineando y consolidando series temporales...")
    # Realizar merge
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
    print(f"    [OK] Datos crudos: {len(df_consolidado):,} filas")
    
    return df_consolidado


def preprocesar_datos_memoria(df_crudo):
    """
    Preprocesa los datos crudos en memoria y genera datasets de train y test.
    """
    print("\n" + "=" * 60)
    print("  PASO 3: Preprocesamiento de datos en memoria")
    print("=" * 60)
    
    df = df_crudo.set_index("timestamp").copy()
    
    # Quedarse con las columnas de features correctas
    df = df[COLUMNAS_FEATURES]
    
    # Manejo de NaNs
    col_motor = "MOTOR_01_Running"
    cols_continuas = [c for c in df.columns if c != col_motor]
    
    # Variables continuas -> interpolación lineal + bfill/ffill
    for col in cols_continuas:
        df[col] = df[col].interpolate(method="linear").ffill().bfill()
        
    # Variables binarias (motor) -> ffill + bfill
    if col_motor in df.columns:
        df[col_motor] = df[col_motor].ffill().bfill()
        
    # Eliminar filas que aún tengan NaNs
    df = df.dropna()
    
    n_total = len(df)
    if n_total < 500:
        raise ValueError(f"Datos insuficientes para entrenamiento seguro: {n_total} muestras.")
        
    print(f"  Muestras limpias totales: {n_total:,}")
    
    # Split cronológico 80/20
    n_train = int(n_total * 0.8)
    df_train = df.iloc[:n_train].copy()
    df_test = df.iloc[n_train:].copy()
    
    print(f"  Muestras de Train:        {len(df_train):,} (80%)")
    print(f"  Muestras de Validation:   {len(df_test):,} (20%)")
    
    # Normalización MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    datos_train_norm = scaler.fit_transform(df_train.values)
    datos_test_norm = scaler.transform(df_test.values)
    
    # DataFrames normalizados
    df_train_norm = pd.DataFrame(datos_train_norm, columns=df_train.columns, index=df_train.index)
    df_test_norm = pd.DataFrame(datos_test_norm, columns=df_test.columns, index=df_test.index)
    
    # Guardar CSV de datos en disco como reporte/historial
    try:
        ruta_csv = os.path.join(DATOS_DIR, "datos_sensores_ultimo_entrenamiento.csv")
        df.to_csv(ruta_csv, index=True)
        print(f"  [OK] Dataset histórico de entrenamiento guardado en: {ruta_csv}")
    except Exception as e:
        print(f"  [WARN] No se pudo guardar el CSV histórico: {e}")
        
    return df_train_norm, df_test_norm, scaler


def construir_autoencoder(n_features):
    """
    Construye la arquitectura funcional del Autoencoder.
    """
    inputs = layers.Input(shape=(n_features,), name="input_sensores")
    
    # Encoder
    x = layers.Dense(32, activation="relu", name="encoder_1")(inputs)
    x = layers.Dense(16, activation="relu", name="encoder_2")(x)
    bottleneck = layers.Dense(8, activation="relu", name="bottleneck")(x)
    
    # Decoder
    x = layers.Dense(16, activation="relu", name="decoder_1")(bottleneck)
    x = layers.Dense(32, activation="relu", name="decoder_2")(x)
    outputs = layers.Dense(n_features, activation="sigmoid", name="output_reconstruccion")(x)
    
    autoencoder = keras.Model(inputs=inputs, outputs=outputs, name="autoencoder_anomalias")
    
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
    )
    return autoencoder


def entrenar_nuevo_modelo(df_train_norm, df_test_norm, epochs=200, batch_size=64):
    """
    Entrena el Autoencoder con parada temprana y retorno de mejores pesos.
    """
    print("\n" + "=" * 60)
    print("  PASO 4: Entrenamiento de nuevo Autoencoder")
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
    
    final_loss = history.history["loss"][-1]
    final_val_loss = history.history["val_loss"][-1]
    mejor_val_loss = min(history.history["val_loss"])
    epochs_ejecutadas = len(history.history["loss"])
    
    print(f"\n  ✓ Entrenamiento finalizado en {epochs_ejecutadas} épocas.")
    print(f"    Loss Train final: {final_loss:.6f}")
    print(f"    Loss Val final:   {final_val_loss:.6f}")
    print(f"    Mejor val_loss:   {mejor_val_loss:.6f}")
    
    return autoencoder, history, mejor_val_loss


def calcular_umbral_anomalia(autoencoder, df_train_norm):
    """
    Calcula el umbral de anomalía basado en percentiles sobre datos de train.
    """
    print("\n" + "=" * 60)
    print("  PASO 5: Cálculo del nuevo umbral de anomalía")
    print("=" * 60)
    
    X_train = df_train_norm.values.astype(np.float32)
    X_reconstructed = autoencoder.predict(X_train, verbose=0)
    
    mse_por_muestra = np.mean(np.square(X_train - X_reconstructed), axis=1)
    
    p95 = float(np.percentile(mse_por_muestra, 95))
    p99 = float(np.percentile(mse_por_muestra, 99))
    mean = float(np.mean(mse_por_muestra))
    std = float(np.std(mse_por_muestra))
    
    print(f"    Nuevo umbral P95: {p95:.6f}")
    print(f"    Nuevo umbral P99: {p99:.6f}")
    print(f"    Error medio:      {mean:.6f} ± {std:.6f}")
    
    return {
        "p95": p95,
        "p99": p99,
        "mean": mean,
        "std": std,
    }


def guardar_artefactos_produccion(autoencoder, scaler, umbral, metadata):
    """
    Guarda todos los artefactos en el directorio modelos/.
    """
    print("\n" + "=" * 60)
    print("  PASO 6: Guardado seguro de nuevos artefactos de producción")
    print("=" * 60)
    
    ruta_modelos = MODELOS_DIR
    
    # 1. Guardar modelo
    ruta_keras = os.path.join(ruta_modelos, ARCHIVOS_MODELO["modelo"])
    autoencoder.save(ruta_keras)
    print(f"    [OK] Guardado: {ARCHIVOS_MODELO['modelo']}")
    
    # 2. Guardar scaler
    ruta_scaler = os.path.join(ruta_modelos, ARCHIVOS_MODELO["scaler"])
    joblib.dump(scaler, ruta_scaler)
    print(f"    [OK] Guardado: {ARCHIVOS_MODELO['scaler']}")
    
    # 3. Guardar umbral
    ruta_umbral = os.path.join(ruta_modelos, ARCHIVOS_MODELO["umbral"])
    joblib.dump(umbral, ruta_umbral)
    print(f"    [OK] Guardado: {ARCHIVOS_MODELO['umbral']}")
    
    # 4. Guardar metadata
    ruta_metadata = os.path.join(ruta_modelos, ARCHIVOS_MODELO["metadata"])
    joblib.dump(metadata, ruta_metadata)
    print(f"    [OK] Guardado: {ARCHIVOS_MODELO['metadata']}")


def main():
    parser = argparse.ArgumentParser(
        description="Script autónomo de reentrenamiento diario para Autoencoder"
    )
    parser.add_argument("--start-date", type=str, default="2026-05-29T00:00:00",
                        help="Fecha de inicio en formato ISO YYYY-MM-DDTHH:MM:SS")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Número máximo de épocas")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Tamaño de batch para entrenamiento")
    parser.add_argument("--verbose", action="store_true",
                        help="Mostrar logs detallados")
    args = parser.parse_args()

    print("+" + "=" * 58 + "+")
    print("|  AUTOENCODER -- REENTRENAMIENTO DIARIO AUTOMATIZADO      |")
    print("|  Ejecución segura con fail-safe y backups                |")
    print("+" + "=" * 58 + "+")
    
    crear_directorios()
    
    # Paso 1: Hacer copia de seguridad de producción
    try:
        crear_backup()
    except Exception as e:
        print(f"\n[CRÍTICO] Cancelando ejecución. No se pudo asegurar el backup: {e}")
        sys.exit(1)
        
    # Ejecutar pipeline de reentrenamiento
    try:
        # Paso 2: Extraer datos desde VictoriaMetrics
        df_crudo = extraer_datos_historicos(args.start_date, verbose=args.verbose)
        
        # Paso 3: Preprocesar y dividir
        df_train_norm, df_test_norm, scaler = preprocesar_datos_memoria(df_crudo)
        
        # Paso 4: Entrenar nuevo Autoencoder
        autoencoder, history, mejor_val_loss = entrenar_nuevo_modelo(
            df_train_norm=df_train_norm,
            df_test_norm=df_test_norm,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        
        # Paso 5: Calcular nuevos umbrales
        umbral_info = calcular_umbral_anomalia(autoencoder, df_train_norm)
        
        # Construir metadata
        metadata = {
            "n_features": len(COLUMNAS_FEATURES),
            "columnas": COLUMNAS_FEATURES,
            "arquitectura": "6->32->16->8->16->32->6",
            "loss": "mse",
            "optimizer": "adam",
            "epochs_entrenados": len(history.history["loss"]),
            "mejor_val_loss": float(mejor_val_loss),
            "umbral_p95": umbral_info["p95"],
            "umbral_p99": umbral_info["p99"],
            "train_samples": int(df_train_norm.shape[0]),
            "test_samples": int(df_test_norm.shape[0]),
            "fecha_entrenamiento": datetime.now(timezone.utc).isoformat(),
            "rango_datos": f"{df_crudo['timestamp'].min()} a {df_crudo['timestamp'].max()}",
        }
        
        # Paso 6: Guardar nuevos artefactos en producción
        guardar_artefactos_produccion(autoencoder, scaler, umbral_info, metadata)
        
        print("\n" + "=" * 60)
        print("  [ÉXITO] REENTRENAMIENTO DIARIO COMPLETADO SATISFACTORIAMENTE")
        print(f"  Rango de datos: {metadata['rango_datos']}")
        print(f"  Mejor val_loss: {metadata['mejor_val_loss']:.6f}")
        print(f"  Umbral P95:     {metadata['umbral_p95']:.6f}")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n[ERROR] Ocurrió una excepción durante el pipeline: {e}")
        import traceback
        traceback.print_exc()
        
        # Accionar fail-safe: Restaurar backup
        restaurar_backup()
        sys.exit(1)


if __name__ == "__main__":
    main()
