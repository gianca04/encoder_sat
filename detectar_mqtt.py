# -*- coding: utf-8 -*-
"""
===============================================================================
detectar_mqtt.py — Detección de Anomalías en Tiempo Real vía MQTT
===============================================================================

PROPÓSITO:
    Este script se suscribe a los topics del broker MQTT en tiempo real,
    recolecta muestras completas, normaliza los valores y utiliza el modelo
    Autoencoder activo en producción para clasificar anomalías al instante.

FUNCIONAMIENTO:
    - Se suscribe a los topics indicados en TOPIC_MAP.
    - Acumula los datos entrantes en un buffer multivariado ordenado.
    - Cuando se dispone de al menos un valor de cada sensor, procesa la
      muestra con el scaler del modelo y realiza la inferencia.
    - Si el Error de Reconstrucción (MSE) supera el umbral (P95 o P99),
      emite una alerta detallando qué sensor contribuyó más al error.

USO:
    python detectar_mqtt.py
    python detectar_mqtt.py --verbose
    python detectar_mqtt.py --umbral p99
    python detectar_mqtt.py --intervalo 15

===============================================================================
"""

import os
import sys
import json
import time
import signal
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from collections import OrderedDict

# Forzar UTF-8 en la consola de Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import paho.mqtt.client as mqtt

from utils import (
    cargar_modelo,
    cargar_scaler,
    cargar_umbral,
    crear_directorios,
    SENSORES,
    COLUMNAS_FEATURES,
    MODELOS_DIR,
    DATOS_DIR,
)

# Configuración de Conexión MQTT
MQTT_BROKER = os.getenv("MQTT_BROKER", "192.168.10.33")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_USER = os.getenv("MQTT_USER", "sat_lab")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "&HjVFmrhuBK")

# Mapping de Topics MQTT a Columnas de Features del Autoencoder
TOPIC_MAP = {
    "lab_sat/FIT_001/Flow":     "FIT_001_Flow",
    "lab_sat/LIT_001/Level":    "LIT_001_Level",
    "lab_sat/LIT_002/Level":    "LIT_002_Level",
    "lab_sat/MOTOR_01/Running": "MOTOR_01_Running",
    "lab_sat/TT_001/Temp":      "TT_001_Temp",
    "lab_sat/TT_002/Temp":      "TT_002_Temp",
}

COLUMNA_TO_TOPIC = {v: k for k, v in TOPIC_MAP.items()}

# Constantes ISO 13374 AHI (se mantiene variable para no romper compatibilidad MQTT)
ORDEN_NAMUR = {"OPTIMAL": 0, "ACCEPTABLE": 1, "DEGRADED": 2, "CRITICAL": 3}
NAMUR_INVERSO = {v: k for k, v in ORDEN_NAMUR.items()}

# Umbral de frescura de dato (segundos) para quality code
STALE_TIMEOUT_S = 120


class DetectorAnomaliasMQTT:
    def __init__(self, umbral_tipo="p95", verbose=False, intervalo_eval=15):
        self.umbral_tipo = umbral_tipo
        self.verbose = verbose
        self.intervalo_eval = intervalo_eval

        # Buffer multivariado inicializado en None
        self.buffer = OrderedDict()
        self.buffer_timestamps = OrderedDict()
        for col in COLUMNAS_FEATURES:
            self.buffer[col] = None
            self.buffer_timestamps[col] = None

        self.ultima_eval = 0
        self.n_evaluaciones = 0
        self.n_anomalias = 0
        self.mensajes_recibidos = 0
        self.running = True

        self._cargar_artefactos()
        self.log_anomalias = []

    def _cargar_artefactos(self):
        """Carga de manera dinámica el modelo y escaladores activos."""
        print("\n  Cargando artefactos del modelo...")
        print("  " + "-" * 50)

        self.autoencoder = cargar_modelo()
        self.scaler = cargar_scaler()
        self.umbral_info = cargar_umbral()
        self.umbral_valor = self.umbral_info[self.umbral_tipo]

        # Cargar metadata
        ruta_metadata = os.path.join(MODELOS_DIR, "metadata_modelo.joblib")
        import joblib
        self.metadata = joblib.load(ruta_metadata)
        self.columnas = self.metadata["columnas"]

        # Cargar estadísticas por sensor (si existen del entrenamiento)
        self.stats_por_sensor = self.umbral_info.get("por_sensor", None)
        if self.stats_por_sensor:
            print(f"\n  [OK] Estadísticas por sensor cargadas (NAMUR NE107)")
            for col in self.columnas:
                s = self.stats_por_sensor.get(col, {})
                print(f"    {col:25s}: mean={s.get('mean',0):.6f} std={s.get('std',0):.6f} p95={s.get('p95',0):.6f}")
        else:
            print(f"\n  [WARN] Sin estadísticas por sensor. Reentrenar para habilitar salud NAMUR NE107.")

        print(f"\n  [OK] Modelo listo para inferencia en tiempo real")
        print(f"  Umbral ({self.umbral_tipo}): {self.umbral_valor:.6f}")
        print(f"  Columnas cargadas: {self.columnas}")

    def _evaluar(self):
        ahora = time.time()

        # Respetar frecuencia de evaluación mínima
        if ahora - self.ultima_eval < self.intervalo_eval:
            return

        # Comprobar que el buffer cuente con lecturas completas
        faltantes = [k for k, v in self.buffer.items() if v is None]
        if faltantes:
            if self.verbose:
                print(f"  [WAIT] Esperando datos de sensores: {faltantes}")
            return

        self.ultima_eval = ahora
        self.n_evaluaciones += 1

        # Construir muestra alineada
        valores = np.array([[self.buffer[col] for col in self.columnas]])

        # Normalizar e inferir
        valores_norm = self.scaler.transform(valores)
        reconstruido_norm = self.autoencoder.predict(valores_norm, verbose=0)

        # Medir errores cuadráticos por feature y MSE total
        errores_features = np.square(valores_norm - reconstruido_norm)[0]
        mse_total = np.mean(errores_features)

        es_anomalia = mse_total > self.umbral_valor
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")

        # ── INVERSE TRANSFORM PARA VALORES ESPERADOS (SIEMPRE, NO SOLO EN ANOMALÍA) ──
        reconstruido_raw = self.scaler.inverse_transform(reconstruido_norm)[0]

        # ── IDENTIFICAR SENSOR CRÍTICO ──
        idx_peor = np.argmax(errores_features)
        peor_sensor = self.columnas[idx_peor]

        # ── CONSTRUIR MÉTRICAS POR SENSOR (PUBLICACIÓN CONTINUA) ──
        desviacion_ratio = float(mse_total / self.umbral_valor)
        lista_sensores = []
        ahora_ts = time.time()
        
        for i, col in enumerate(self.columnas):
            topic_orig = COLUMNA_TO_TOPIC.get(col, "")
            parts_orig = topic_orig.split("/") if topic_orig else []
            eq_id = parts_orig[1] if len(parts_orig) > 1 else col
            met_name = parts_orig[2] if len(parts_orig) > 2 else "Valor"

            # Metadata del sensor desde la configuración central
            sensor_cfg = SENSORES.get(col, {})

            # ── QUALITY CODE basado en frescura del dato ──
            ts_dato = self.buffer_timestamps.get(col)
            if ts_dato is None:
                quality = "BAD"
                ts_dato_iso = None
            elif ahora_ts - ts_dato > STALE_TIMEOUT_S:
                quality = "STALE"
                ts_dato_iso = datetime.fromtimestamp(ts_dato, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
            else:
                quality = "GOOD"
                ts_dato_iso = datetime.fromtimestamp(ts_dato, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")

            # ── SALUD POR SENSOR: NAMUR NE107 con Z-Score Individual ──
            if self.stats_por_sensor and col in self.stats_por_sensor:
                stats = self.stats_por_sensor[col]
                sensor_mean = stats["mean"]
                sensor_std = stats["std"] if stats["std"] > 1e-10 else 1e-10
                sensor_p95 = stats["p95"]
                sensor_p99 = stats["p99"]

                z_score = (errores_features[i] - sensor_mean) / sensor_std
                salud_sensor = float(max(0.0, min(100.0, 100.0 * (1.0 - z_score / 6.0))))

                if salud_sensor >= 85.0:
                    estado_sensor = "OPTIMAL"
                elif salud_sensor >= 70.0:
                    estado_sensor = "ACCEPTABLE"
                elif salud_sensor >= 50.0:
                    estado_sensor = "DEGRADED"
                else:
                    estado_sensor = "CRITICAL"

                # Anomalía INDIVIDUAL basada en umbral propio del sensor
                es_sensor_anomalo = bool(errores_features[i] > sensor_p95)
            else:
                salud_sensor = float(100.0 * (1.0 - (errores_features[i] / (self.umbral_valor * 5.0))))
                salud_sensor = max(0.0, min(100.0, salud_sensor))
                estado_sensor = "OPTIMAL" if salud_sensor >= 85.0 else ("ACCEPTABLE" if salud_sensor >= 70.0 else ("DEGRADED" if salud_sensor >= 50.0 else "CRITICAL"))
                es_sensor_anomalo = bool(errores_features[i] > self.umbral_valor)

            valor_actual = float(self.buffer[col])
            valor_esperado = float(reconstruido_raw[i])

            lista_sensores.append({
                "equipo_id": eq_id,
                "metrica": met_name,
                "unidad": sensor_cfg.get("unidad", ""),
                "tipo_dato": sensor_cfg.get("tipo", "REAL"),
                "valor_actual": round(valor_actual, 4),
                "valor_esperado": round(valor_esperado, 4),
                "error_absoluto": round(abs(valor_actual - valor_esperado), 4),
                "error_reconstruccion": round(float(errores_features[i]), 6),
                "salud_pct": round(salud_sensor, 2),
                "estado_namur": estado_sensor,
                "estado_code": ORDEN_NAMUR.get(estado_sensor, 0),
                "es_sensor_anomalo": es_sensor_anomalo,
                "es_critico": bool(es_anomalia and i == idx_peor),
                "quality": quality,
                "timestamp_dato": ts_dato_iso,
            })

        # ── ESTADO GENERAL DE PLANTA (Principio del eslabón más débil) ──
        salud_planta = min(sensor["salud_pct"] for sensor in lista_sensores)
        peor_estado_idx = max(ORDEN_NAMUR.get(sensor["estado_namur"], 0) for sensor in lista_sensores)
        estado_planta = NAMUR_INVERSO[peor_estado_idx]
        n_sensores_degradados = sum(1 for sensor in lista_sensores if sensor["estado_namur"] != "OPTIMAL")

        # ── PUBLICAR MÉTRICAS ML EN TOPIC ESTRUCTURADO (CADA EVALUACIÓN) ──
        topic_metricas = "lab_sat/autoencoder/metricas"
        payload_metricas = json.dumps({
            "schema_version": "2.0",
            "timestamp": ts,
            "modelo": {
                "id": "autoencoder_anomalias",
                "umbral_tipo": self.umbral_tipo,
                "umbral_valor": round(float(self.umbral_valor), 6),
            },
            "planta": {
                "salud_pct": round(salud_planta, 2),
                "estado_namur": estado_planta,
                "estado_code": peor_estado_idx,
                "sensores_degradados": n_sensores_degradados,
                "sensores_total": len(lista_sensores),
                "mse": round(float(mse_total), 6),
                "desviacion_ratio": round(desviacion_ratio, 4),
            },
            "sensores": lista_sensores,
        }, ensure_ascii=False)

        try:
            if hasattr(self, "client") and self.client:
                self.client.publish(topic_metricas, payload_metricas, qos=1)
                if self.verbose:
                    print(f"    [MQTT] Métricas ML publicadas en: {topic_metricas}")
                    print(f"    [MQTT] Planta: salud={salud_planta:.1f}% estado={estado_planta} degradados={n_sensores_degradados}/{len(lista_sensores)}")
        except Exception as e:
            print(f"    [WARN] No se pudo publicar métricas ML: {e}")

        # ── BLOQUE DE ANOMALÍA: SOLO LOGGING Y CONSOLA ──
        if es_anomalia:
            self.n_anomalias += 1

            # Extraer info del sensor crítico para la consola
            topic_original = COLUMNA_TO_TOPIC.get(peor_sensor)
            parts = topic_original.split("/") if topic_original else []
            equipo_id = parts[1] if len(parts) > 1 else peor_sensor
            metrica_name = parts[2] if len(parts) > 2 else "Valor"

            print(f"\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"  !! ALERTA: ANOMALÍA DETECTADA EN SENSADO  #{self.n_anomalias}")
            print(f"  !! Timestamp:        {ts}")
            print(f"  !! MSE Muestra:      {mse_total:.6f} (Umbral: {self.umbral_valor:.6f})")
            print(f"  !! Desviación:       {desviacion_ratio:.1f}x sobre el límite")
            print(f"  !! Sensor Crítico:   {peor_sensor} (Equipo: {equipo_id} | Métrica: {metrica_name})")
            print(f"  !!   Valor Actual:   {self.buffer[peor_sensor]:.4f}")
            print(f"  !!   Valor Esperado: {reconstruido_raw[idx_peor]:.4f}")
            print(f"  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

            print(f"  Valores del vector de sensores:")
            for i, col in enumerate(self.columnas):
                marca = " <⚠>" if i == idx_peor else ""
                print(f"    {col:22s}: real={self.buffer[col]:8.4f} esperado={reconstruido_raw[i]:8.4f} error_feat={errores_features[i]:.6f}{marca}")

            self.log_anomalias.append({
                "timestamp": ts,
                "mse_total": float(mse_total),
                "equipo_id": equipo_id,
                "metrica_name": metrica_name,
                "valores": dict(self.buffer),
                "errores": {col: float(errores_features[idx]) for idx, col in enumerate(self.columnas)},
            })
        else:
            if self.verbose:
                print(f"  [{ts}] Lectura normal - MSE={mse_total:.6f} ({mse_total/self.umbral_valor*100:.0f}% del umbral)")
            else:
                if self.n_evaluaciones % 10 == 0:
                    print(f"  [{ts}] Operación normal | Eval #{self.n_evaluaciones} | Anomalías: {self.n_anomalias}")

    # callbacks MQTT
    def on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            print(f"\n  [OK] Conexión establecida con broker MQTT: {MQTT_BROKER}:{MQTT_PORT}")
            print(f"  Suscribiéndose a topics de sensores...")
            for topic, col in TOPIC_MAP.items():
                client.subscribe(topic, qos=1)
                print(f"    -> {topic} ({col})")
            # Publicar estado ONLINE (sobrescribe el LWT OFFLINE)
            client.publish("lab_sat/autoencoder/status",
                json.dumps({"status": "ONLINE", "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")}),
                qos=1, retain=True)
            print(f"\n  [OK] Escuchando lecturas industriales...")
        else:
            print(f"  [ERROR] Fallo de autenticación o conexión MQTT. Código: {rc}")

    def on_disconnect(self, client, userdata, rc, properties=None):
        if rc != 0:
            print(f"  [WARN] Desconexión imprevista de MQTT. Reconectando...")
        else:
            print(f"  [INFO] Conexión MQTT cerrada voluntariamente.")

    def on_message(self, client, userdata, msg):
        self.mensajes_recibidos += 1
        topic = msg.topic
        
        try:
            payload = msg.payload.decode("utf-8").strip()
            # Parser JSON
            try:
                data = json.loads(payload)
                if isinstance(data, dict):
                    valor = float(data.get("value", data.get("valor", data.get("v", payload))))
                else:
                    valor = float(data)
            except (json.JSONDecodeError, TypeError):
                valor = float(payload)
        except Exception:
            return

        columna = TOPIC_MAP.get(topic)
        if columna is None:
            # Tolerancia a prefijos en topics
            for t, c in TOPIC_MAP.items():
                if topic.endswith(t.split("/", 1)[-1] if "/" in t else t):
                    columna = c
                    break

        if columna is not None:
            self.buffer[columna] = valor
            self.buffer_timestamps[columna] = time.time()
            if self.verbose:
                print(f"  [LECTURA] {columna} = {valor:.4f}")
            self._evaluar()

    def iniciar(self):
        import platform

        client_id = f"autoencoder_{platform.node()}_{os.getpid()}"
        try:
            client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id=client_id
            )
        except AttributeError:
            client = mqtt.Client(client_id=client_id)

        self.client = client
        client.username_pw_set(MQTT_USER, MQTT_PASSWORD)

        # LWT: si el detector muere, el broker publica estado OFFLINE automáticamente
        client.will_set(
            "lab_sat/autoencoder/status",
            payload=json.dumps({
                "status": "OFFLINE",
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                "reason": "unexpected_disconnect"
            }),
            qos=1,
            retain=True
        )

        client.on_connect = self.on_connect
        client.on_disconnect = self.on_disconnect
        client.on_message = self.on_message
        client.reconnect_delay_set(min_delay=1, max_delay=120)

        try:
            client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            client.loop_forever()
        except KeyboardInterrupt:
            print("\n  Cerrando detector MQTT...")
            self.running = False
        finally:
            try:
                client.disconnect()
            except Exception:
                pass
            self._guardar_reporte()

    def _guardar_reporte(self):
        print(f"\n" + "=" * 60)
        print(f"  REPORTE FINAL DETECTOR MQTT")
        print(f"=" * 60)
        print(f"  Lecturas recibidas:   {self.mensajes_recibidos:,}")
        print(f"  Evaluaciones:         {self.n_evaluaciones:,}")
        print(f"  Anomalías:            {self.n_anomalias:,}")
        
        if self.log_anomalias:
            ruta = os.path.join(DATOS_DIR, "log_anomalias_detectadas_mqtt.json")
            with open(ruta, "w", encoding="utf-8") as f:
                json.dump(self.log_anomalias, f, indent=2, ensure_ascii=False)
            print(f"  Registro guardado en: {ruta}")
        print(f"=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MQTT Real-time Anomaly Detector")
    parser.add_argument("--umbral", type=str, default="p95", choices=["p95", "p99"])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--intervalo", type=int, default=15)
    args = parser.parse_args()

    crear_directorios()
    detector = DetectorAnomaliasMQTT(
        umbral_tipo=args.umbral,
        verbose=args.verbose,
        intervalo_eval=args.intervalo
    )
    detector.iniciar()
