# -*- coding: utf-8 -*-
"""
===============================================================================
test_salud_planta.py — Test de detección de anomalías en la SALUD DE LA PLANTA
===============================================================================

PROPÓSITO:
    Verificar de extremo a extremo (sin tocar InfluxDB, MQTT ni los artefactos
    de producción en modelos/) que el pipeline detecta correctamente degradación
    del estado de salud de toda la planta del laboratorio:

        Data Acquisition -> State Detection -> Health Assessment (ISO 13374)
        Severidad por sensor y de planta                     (NAMUR NE 107)

CÓMO FUNCIONA:
    1. Simula una planta de laboratorio con los 16 tags reales (SENSORES) y sus
       relaciones físicas (flujo ~ RPM, niveles anticorrelados, compresor ~
       presión de aire, etc.), incluyendo bloques de máquina apagada.
    2. Aplica las reglas del proyecto: filtro de régimen operativo (bomba o
       compresor en marcha), split cronológico, RobustScaler ajustado SOLO con
       train y umbrales P95/P98/P99 sobre validación (calcular_umbral de entrenar.py).
    3. Entrena en memoria el MISMO autoencoder LSTM de entrenar.py.
    4. Ejercita DetectorAnomaliasMQTT.evaluar_muestra() (la lógica real de
       producción) con:
         - operación normal  -> tasa de falsas alarmas baja, planta OPTIMAL
         - fallas inyectadas -> detección, severidad y sensor causa raíz

USO:
    python test_salud_planta.py
    python test_salud_planta.py --muestras 100 --epocas 60

SALIDA:
    Tabla de resultados y código de salida 0 (todo OK) / 1 (algún criterio falla).

===============================================================================
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from sklearn.preprocessing import RobustScaler

from utils import SENSORES, COLUMNAS_FEATURES
from entrenar import construir_autoencoder_lstm, calcular_umbral
from detectar_mqtt import DetectorAnomaliasMQTT, ORDEN_NAMUR

SEED = 42

# Criterios de aceptación
MAX_FALSAS_ALARMAS = 0.15       # P95 entrenado -> esperado ~5-10% en holdout
MIN_SALUD_MEDIANA_NORMAL = 85.0  # planta OPTIMAL en operación normal
MIN_DETECCION_FALLA = 0.90       # % de muestras con falla marcadas como anomalía
MIN_PLANTA_NO_OPTIMAL = 0.90     # % de muestras con falla cuya planta != OPTIMAL
MIN_CAUSA_RAIZ_TOP3 = 0.70       # % de muestras con falla donde la causa está en el top-3

# Fallas de un solo sensor cuya severidad depende del estado del proceso: LIT_001 clavado
# en 5% es una desviación pequeña cuando el tanque ya está cerca de 30-35%, y el MSE global
# la diluye entre 16 variables. Se exige 80% (limitación conocida del promedio global).
MIN_DETECCION_ESPECIAL = {"Nivel inconsistente (LIT_001 clavado 5%)": 0.80}


# =============================================================================
# SIMULADOR DE PLANTA
# =============================================================================

def simular_planta(n, rng, con_paradas=True):
    """
    Genera n muestras (1 cada ~15 s) de operación normal de la planta.
    Relaciones físicas simuladas:
      - FIT_001_MAS ~ 0.0035 * RPM (bomba) ; FIT_001_VOL = masa / densidad
      - LIT_001 baja y LIT_002 sube (trasvase entre tanques), sum ~ 100 %
      - Compresor cicla ON/OFF y la presión de aire sube/baja en consecuencia
      - Val_001 y Val_002 abiertas durante el trasvase
    """
    t = np.arange(n)

    # Consigna de velocidad escalonada
    seg = 300
    niveles = np.array([1200.0, 1500.0, 1750.0, 2000.0])
    ref = np.repeat(rng.choice(niveles, size=n // seg + 2), seg)[:n]
    rpm = ref + rng.normal(0, 5, n)

    flujo = 0.0035 * rpm + rng.normal(0, 0.01, n)
    dens = 998.0 + rng.normal(0, 0.3, n)
    temp_f = 25.0 + 1.5 * np.sin(2 * np.pi * t / n * 1.3) + rng.normal(0, 0.05, n)
    vol = flujo / dens + rng.normal(0, 1e-6, n)

    ciclo = 1200
    frac = (t % ciclo) / ciclo
    lit1 = 80.0 - 50.0 * frac + rng.normal(0, 0.15, n)
    lit2 = 20.0 + 50.0 * frac + rng.normal(0, 0.15, n)
    tt = 22.0 + 1.5 * np.sin(2 * np.pi * t / 2000.0) + rng.normal(0, 0.05, n)

    # Compresor: ciclo ON/OFF y presión de aire asociada (sierra 80-100 PSI)
    per = 400
    comp = ((t % per) < 250).astype(float)
    pit = np.empty(n)
    p = 90.0
    for i in range(n):
        p += 0.08 if comp[i] == 1 else -0.05
        p = min(100.0, max(80.0, p))
        pit[i] = p
    pit = pit + rng.normal(0, 0.1, n)

    df = pd.DataFrame({
        "PIT_001": pit,
        "FIT_001_MAS": flujo,
        "FIT_001_DENS": dens,
        "FIT_001_TEMP": temp_f,
        "FIT_001_VOL": vol,
        "LIT_001": lit1,
        "LIT_002": lit2,
        "TT_001": tt,
        "Bomba_Agua_001_STATUS": np.ones(n),
        "Bomba_Agua_001_REF": rpm,
        "Val_001": np.ones(n),
        "Val_002": np.ones(n),
        "Val_003": np.zeros(n),
        "Val_004": np.zeros(n),
        "Mot_Comp_001": comp,
        # MOTOR_01 NO está unido al proceso: ruido independiente
        "MOTOR_01": (rng.random(n) > 0.5).astype(float),
    })

    if con_paradas:
        # Bloques de máquina apagada (bomba y compresor en 0): régimen distinto
        for ini in (int(n * 0.10), int(n * 0.55)):
            fin = ini + 200
            df.loc[ini:fin, "Bomba_Agua_001_STATUS"] = 0.0
            df.loc[ini:fin, "Mot_Comp_001"] = 0.0
            df.loc[ini:fin, "Bomba_Agua_001_REF"] = 0.0
            df.loc[ini:fin, ["FIT_001_MAS", "FIT_001_VOL"]] = 0.0
            df.loc[ini:fin, ["Val_001", "Val_002"]] = 0.0

    return df[COLUMNAS_FEATURES]


# =============================================================================
# FALLAS INYECTADAS: (nombre, función, sensores causa raíz aceptables)
# =============================================================================

def f_bloqueo_flujo(df):
    df["FIT_001_MAS"] = 0.0
    df["FIT_001_VOL"] = 0.0
    return df

def f_fuga_aire(df):
    df["PIT_001"] = 40.0
    return df

def f_sobretemperatura(df):
    df["FIT_001_TEMP"] = df["FIT_001_TEMP"] + 12.0
    return df

def f_nivel_inconsistente(df):
    df["LIT_001"] = 5.0
    return df

def f_valvula_incoherente(df):
    df["Val_001"] = 0.0
    df["Val_002"] = 0.0
    return df

def f_falla_multiple(df):
    df["PIT_001"] = 55.0
    df["FIT_001_MAS"] = df["FIT_001_MAS"] * 0.3
    df["FIT_001_TEMP"] = df["FIT_001_TEMP"] + 9.0
    df["LIT_002"] = 95.0
    return df

FALLAS = [
    ("Bloqueo de flujo (bomba gira, flujo=0)",   f_bloqueo_flujo,     {"FIT_001_MAS", "FIT_001_VOL", "Bomba_Agua_001_REF"}),
    ("Fuga aire instrumentación (PIT 40 PSI)",   f_fuga_aire,         {"PIT_001", "Mot_Comp_001"}),
    ("Sobretemperatura fluido (+12 °C)",         f_sobretemperatura,  {"FIT_001_TEMP", "FIT_001_DENS"}),
    ("Nivel inconsistente (LIT_001 clavado 5%)", f_nivel_inconsistente, {"LIT_001", "LIT_002"}),
    ("Válvulas cerradas con flujo activo",       f_valvula_incoherente, {"Val_001", "Val_002", "FIT_001_MAS", "FIT_001_VOL"}),
    ("Falla múltiple (4 sensores)",              f_falla_multiple,    {"PIT_001", "FIT_001_MAS", "FIT_001_TEMP", "LIT_002"}),
]


# =============================================================================
# CONSTRUCCIÓN DEL DETECTOR (lógica real, sin broker MQTT)
# =============================================================================

def entrenar_en_memoria(n_total, epocas, verbose=False):
    """Aplica las reglas del proyecto y devuelve (detector, df_holdout)."""
    rng = np.random.default_rng(SEED)
    tf.keras.utils.set_random_seed(SEED)

    df = simular_planta(n_total, rng)

    # Regla 1.1: solo régimen operativo (bomba o compresor en marcha)
    mask = (df["Bomba_Agua_001_STATUS"] == 1) | (df["Mot_Comp_001"] == 1)
    n_apagado = int((~mask).sum())
    df = df[mask].reset_index(drop=True)
    print(f"  Datos simulados: {n_total:,} | descartados por máquina apagada: {n_apagado:,} | útiles: {len(df):,}")

    # Regla 1.3: split cronológico train / validación / holdout (70/15/15).
    #   train   -> pesos y scaler
    #   val     -> early stopping + umbrales (regla 3.1)
    #   holdout -> evaluación del test (nunca visto)
    n_train = int(len(df) * 0.70)
    n_val = int(len(df) * 0.85)
    df_train, df_val, df_hold = df.iloc[:n_train], df.iloc[n_train:n_val], df.iloc[n_val:]

    scaler = RobustScaler()
    Xtr = scaler.fit_transform(df_train.values).astype(np.float32)
    Xva = scaler.transform(df_val.values).astype(np.float32)

    modelo = construir_autoencoder_lstm(Xtr.shape[1])
    modelo.fit(
        Xtr, Xtr, validation_data=(Xva, Xva), epochs=epocas, batch_size=128,
        shuffle=True, verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
    )

    df_val_norm = pd.DataFrame(Xva, columns=df.columns)
    umbral_info = calcular_umbral(modelo, df_val_norm)

    # Detector real, saltando __init__ (que carga artefactos de producción y no
    # conecta a MQTT). Se inyecta el modelo entrenado en memoria.
    det = object.__new__(DetectorAnomaliasMQTT)
    det.umbral_tipo = "p95"
    det.verbose = False
    det.autoencoder = modelo
    det.scaler = scaler
    det.umbral_info = umbral_info
    det.umbral_valor = umbral_info["p95"]
    det.columnas = list(COLUMNAS_FEATURES)
    det.stats_por_sensor = umbral_info["por_sensor"]
    det.buffer = {}
    det.buffer_timestamps = {}
    det.topic_por_columna = {c: f"sat_lab/telemetry/nodo_test/dev_test/{c}" for c in COLUMNAS_FEATURES}
    return det, df_hold.reset_index(drop=True), n_apagado


def evaluar_df(det, df):
    """Evalúa fila a fila con el núcleo real de producción."""
    resultados = []
    for _, fila in df.iterrows():
        ahora = time.time()
        for c in det.columnas:
            det.buffer[c] = float(fila[c])
            det.buffer_timestamps[c] = ahora
        resultados.append(det.evaluar_muestra())
    return resultados


def resumir(resultados, causas_esperadas=None):
    n = len(resultados)
    detectadas = np.mean([r["es_anomalia"] for r in resultados])
    no_optimal = np.mean([r["peor_estado_idx"] > 0 for r in resultados])
    salud = np.array([r["salud_planta"] for r in resultados])
    res = {
        "tasa_anomalia": float(detectadas),
        "planta_no_optimal": float(no_optimal),
        "salud_mediana": float(np.median(salud)),
        "salud_min": float(salud.min()),
        "estado_modal": max(
            {r["estado_planta"] for r in resultados},
            key=lambda e: sum(1 for r in resultados if r["estado_planta"] == e),
        ),
    }
    if causas_esperadas:
        hits1, hits3, hits_s = 0, 0, 0
        for r in resultados:
            orden = np.argsort(r["errores_features"])[::-1]
            top3 = {r["lista_sensores"][i]["tag_name"] for i in orden[:3]}
            hits1 += r["peor_sensor"] in causas_esperadas
            hits3 += len(top3 & causas_esperadas) > 0
            hits_s += any(s["es_sensor_anomalo"] and s["tag_name"] in causas_esperadas
                          for s in r["lista_sensores"])
        res["causa_top1"] = hits1 / n
        res["causa_top3"] = hits3 / n
        res["sensor_flag"] = hits_s / n
    return res


# =============================================================================
# EJECUCIÓN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--muestras", type=int, default=150, help="Muestras evaluadas por escenario")
    ap.add_argument("--total", type=int, default=6000, help="Muestras simuladas totales")
    ap.add_argument("--epocas", type=int, default=80)
    args = ap.parse_args()

    print("=" * 72)
    print("  TEST SALUD DE PLANTA — simulación + autoencoder LSTM en memoria")
    print("=" * 72)

    det, df_test, n_apagado = entrenar_en_memoria(args.total, args.epocas)

    # Muestras de holdout (cronológico, nunca vistas en train)
    rng = np.random.default_rng(SEED + 1)
    idx = np.sort(rng.choice(len(df_test), size=min(args.muestras, len(df_test)), replace=False))
    base = df_test.iloc[idx].reset_index(drop=True)

    fallos = []
    filas = []

    # --- 1. Régimen: la máquina apagada debió descartarse del entrenamiento
    if n_apagado <= 0:
        fallos.append("El filtro de régimen operativo no descartó datos de máquina apagada")

    # --- 2. Operación normal
    print("\n  Evaluando operación normal...")
    res_normal = resumir(evaluar_df(det, base.copy()))
    filas.append(("OPERACIÓN NORMAL", res_normal, None))
    if res_normal["tasa_anomalia"] > MAX_FALSAS_ALARMAS:
        fallos.append(f"Falsas alarmas {res_normal['tasa_anomalia']:.0%} > {MAX_FALSAS_ALARMAS:.0%}")
    if res_normal["salud_mediana"] < MIN_SALUD_MEDIANA_NORMAL:
        fallos.append(f"Salud mediana en normal {res_normal['salud_mediana']:.1f}% < {MIN_SALUD_MEDIANA_NORMAL}%")

    # --- 3. Fallas inyectadas
    for nombre, fn, causas in FALLAS:
        print(f"  Evaluando falla: {nombre}...")
        df_f = fn(base.copy())
        res = resumir(evaluar_df(det, df_f), causas)
        filas.append((nombre, res, causas))
        min_det = MIN_DETECCION_ESPECIAL.get(nombre, MIN_DETECCION_FALLA)
        if res["tasa_anomalia"] < min_det:
            fallos.append(f"[{nombre}] detección {res['tasa_anomalia']:.0%} < {min_det:.0%}")
        if res["planta_no_optimal"] < min(min_det, MIN_PLANTA_NO_OPTIMAL):
            fallos.append(f"[{nombre}] planta no-OPTIMAL {res['planta_no_optimal']:.0%} < {min(min_det, MIN_PLANTA_NO_OPTIMAL):.0%}")
        if res["causa_top3"] < MIN_CAUSA_RAIZ_TOP3:
            fallos.append(f"[{nombre}] causa raíz en top-3 {res['causa_top3']:.0%} < {MIN_CAUSA_RAIZ_TOP3:.0%}")

    # --- 4. Calidad de dato: un sensor sin actualizar debe marcarse STALE (regla 4.1)
    fila = base.iloc[0]
    ahora = time.time()
    for c in det.columnas:
        det.buffer[c] = float(fila[c])
        det.buffer_timestamps[c] = ahora
    det.buffer_timestamps["LIT_001"] = ahora - 10_000
    r = det.evaluar_muestra()
    calidad = {s["tag_name"]: s["quality"] for s in r["lista_sensores"]}
    stale_ok = calidad["LIT_001"] == "STALE" and calidad["PIT_001"] == "GOOD"
    if not stale_ok:
        fallos.append("Sensor sin actualizar no se marcó como STALE")

    # --- Reporte
    print("\n" + "=" * 72)
    print("  RESULTADOS")
    print("=" * 72)
    print(f"  {'Escenario':44s} {'Anom%':>6s} {'NoOpt%':>7s} {'Salud~':>7s} {'Top1':>5s} {'Top3':>5s} {'Sens%':>6s}  Estado")
    print(f"  {'-'*44} {'-'*6} {'-'*7} {'-'*7} {'-'*5} {'-'*5} {'-'*6}  {'-'*10}")
    for nombre, r, causas in filas:
        t1 = f"{r['causa_top1']:.0%}" if "causa_top1" in r else "  -"
        t3 = f"{r['causa_top3']:.0%}" if "causa_top3" in r else "  -"
        sf = f"{r['sensor_flag']:.0%}" if "sensor_flag" in r else "  -"
        print(f"  {nombre:44s} {r['tasa_anomalia']:6.0%} {r['planta_no_optimal']:7.0%} "
              f"{r['salud_mediana']:6.1f}% {t1:>5s} {t3:>5s} {sf:>6s}  {r['estado_modal']}")
    print("\n  Anom%=muestras marcadas anomalía | NoOpt%=planta != OPTIMAL | Salud~=mediana planta")
    print("  Top1/Top3=causa raíz acertada | Sens%=sensor causa marcado individualmente (P95 propio)")
    print(f"\n  Sensor STALE detectado (regla 4.1): {'OK' if stale_ok else 'FALLA'}")
    print(f"  Máquina apagada descartada del train (regla 1.1): {n_apagado:,} muestras")

    print("\n" + "=" * 72)
    if fallos:
        print(f"  RESULTADO: FALLÓ ({len(fallos)} criterio(s))")
        for f in fallos:
            print(f"   - {f}")
        print("=" * 72)
        sys.exit(1)
    print("  RESULTADO: OK — la salud de la planta se evalúa correctamente")
    print("=" * 72)


if __name__ == "__main__":
    main()
