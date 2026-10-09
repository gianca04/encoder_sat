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
from mqtt_publisher import HealthAssessmentPublisher

# Configuración de Conexión MQTT
MQTT_BROKER = os.getenv("MQTT_BROKER", "192.168.10.208")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_USER = os.getenv("MQTT_USER", "sat_lab")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "&HjVFmrhuBK")
MQTT_TOPIC_PREFIX = os.getenv("MQTT_TOPIC_PREFIX", "edge/gateway")
MQTT_TOPIC_SUB = os.getenv("MQTT_TOPIC_SUB", "sat_lab/telemetry/#")
MQTT_CLIENT_ID = os.getenv("MQTT_CLIENT_ID", "sat_lab_backend_worker")
MQTT_TOPIC_NOTIFICATIONS = os.getenv("MQTT_TOPIC_NOTIFICATIONS", "lab_sat/notifications")
MQTT_TOPIC_METRICS = os.getenv("MQTT_TOPIC_METRICS", "lab_sat/metrics")

# Ya no usamos un TOPIC_MAP estático, extraeremos el tag_name, node_id y device_id dinámicamente
# del topic MQTT: sat_lab/telemetry/${node_id}/${device_id}/${tag_name}

# Constantes ISO 13374 AHI (se mantiene variable para no romper compatibilidad MQTT)
ORDEN_NAMUR = {"OPTIMAL": 0, "ACCEPTABLE": 1, "DEGRADED": 2, "CRITICAL": 3}
NAMUR_INVERSO = {v: k for k, v in ORDEN_NAMUR.items()}

# Umbral de frescura de dato (segundos) para quality code
STALE_TIMEOUT_S = 120


def salud_desde_percentiles(valor, p95, p98, p99):
    """
    Mapea un error de reconstrucción a salud % anclada en percentiles de datos
    normales de validación (regla 3.3 / NAMUR NE 107):
        <= P95 -> 100..85  OPTIMAL            (Normal)
        <= P98 -> 85..70   ACCEPTABLE         (Maintenance Required)
        <= P99 -> 70..50   DEGRADED           (Out of Specification)
        >  P99 -> 50..0    CRITICAL           (Failure)   (0% en 2xP99)
    """
    p95 = max(float(p95), 1e-12)
    p98 = max(float(p98), p95)
    p99 = max(float(p99), p98)
    return float(np.interp(valor, [0.0, p95, p98, p99, 2.0 * p99], [100.0, 85.0, 70.0, 50.0, 0.0]))


def estado_desde_salud(salud):
    if salud >= 85.0:
        return "OPTIMAL"
    if salud >= 70.0:
        return "ACCEPTABLE"
    if salud >= 50.0:
        return "DEGRADED"
    return "CRITICAL"



class DetectorAnomaliasMQTT:
    def __init__(self, umbral_tipo="p95", verbose=False, intervalo_eval=15):
        self.umbral_tipo = umbral_tipo
        self.verbose = verbose
        self.intervalo_eval = intervalo_eval

        # Variables para Histéresis / Anti-Flapping (Regla 4.2)
        self.historial_anomalias = []
        self.VENTANA_M = 5
        self.CONFIRMACIONES_N = 3
        self.alarma_activa = False

        # Buffer multivariado inicializado en None
        self.buffer = OrderedDict()
        self.buffer_timestamps = OrderedDict()
        self.topic_por_columna = {} # Para almacenar el último topic real y extraer node/device
        for col in COLUMNAS_FEATURES:
            self.buffer[col] = None
            self.buffer_timestamps[col] = None
            self.topic_por_columna[col] = None

        self.ultima_eval = 0
        self.n_evaluaciones = 0
        self.n_anomalias = 0
        self.mensajes_recibidos = 0
        self.running = True

        self.publisher = None
        self.ultimo_estado_planta = None  # Almacenamiento ligero en memoria para cambios de estado

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

    def evaluar_muestra(self):
        """
        Núcleo puro de inferencia (sin efectos MQTT): toma el buffer multivariado
        actual, normaliza, reconstruye y calcula salud por sensor y de la planta
        (ISO 13374 State Detection -> Health Assessment / NAMUR NE107).
        Es el método que ejercita test_salud_planta.py.
        """
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
            topic_real = self.topic_por_columna.get(col)
            # Extrayendo info del topic sat_lab/telemetry/{node_id}/{device_id}/{tag_name}
            parts_orig = topic_real.split("/") if topic_real else []
            eq_id = parts_orig[-2] if len(parts_orig) >= 2 else col
            node_id = parts_orig[-3] if len(parts_orig) >= 3 else "unknown"

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

            # ── SALUD POR SENSOR: bandas P95/P98/P99 propias (NAMUR NE107, regla 3.3) ──
            stats = (self.stats_por_sensor or {}).get(col)
            if stats:
                p95s, p99s = stats["p95"], stats["p99"]
                p98s = stats.get("p98", (p95s + p99s) / 2.0)
            else:
                p95s, p99s = self.umbral_info["p95"], self.umbral_info["p99"]
                p98s = self.umbral_info.get("p98", (p95s + p99s) / 2.0)

            salud_sensor = salud_desde_percentiles(errores_features[i], p95s, p98s, p99s)
            estado_sensor = estado_desde_salud(salud_sensor)
            # Anomalía INDIVIDUAL basada en umbral propio del sensor
            es_sensor_anomalo = bool(errores_features[i] > p95s)

            valor_actual = float(self.buffer[col])
            valor_esperado = float(reconstruido_raw[i])

            lista_sensores.append({
                "node_id": node_id,
                "device_id": eq_id,
                "tag_name": col,
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

        # ── ESTADO GENERAL DE PLANTA (ISO 13374: sobre el residuo GLOBAL) ──
        # El mínimo de N sensores se degrada por azar al crecer N (falsos positivos);
        # el estado de planta se evalúa sobre el MSE global contra sus percentiles
        # de validación. El detalle por sensor queda para el diagnóstico de causa raíz.
        p95g = self.umbral_info["p95"]
        p99g = self.umbral_info["p99"]
        p98g = self.umbral_info.get("p98", (p95g + p99g) / 2.0)
        salud_planta = salud_desde_percentiles(mse_total, p95g, p98g, p99g)
        estado_planta = estado_desde_salud(salud_planta)
        peor_estado_idx = ORDEN_NAMUR[estado_planta]
        n_sensores_degradados = sum(1 for sensor in lista_sensores if sensor["estado_namur"] != "OPTIMAL")

        return {
            "ts": ts,
            "es_anomalia": bool(es_anomalia),
            "mse_total": float(mse_total),
            "desviacion_ratio": desviacion_ratio,
            "errores_features": errores_features,
            "reconstruido_raw": reconstruido_raw,
            "idx_peor": int(idx_peor),
            "peor_sensor": peor_sensor,
            "lista_sensores": lista_sensores,
            "salud_planta": salud_planta,
            "peor_estado_idx": peor_estado_idx,
            "estado_planta": estado_planta,
            "n_sensores_degradados": n_sensores_degradados,
        }

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

        # ── REGLA OT 1.1: MÁSCARA DE ESTADO OPERATIVO (Golden Baseline) ──
        # Solo evaluamos si el PLC indica que el compresor DEBERÍA estar encendido.
        if "Mot_Comp_001" in self.buffer and self.buffer["Mot_Comp_001"] < 0.5:
            if self.alarma_activa:
                print(f"\n  [INFO] Máquina apagada por comando (Mot_Comp_001=0). Reseteando alarmas.")
                self.alarma_activa = False
                self.historial_anomalias.clear()
            
            if self.verbose:
                print(f"  [{time.strftime('%H:%M:%S')}] Planta detenida (Mot_Comp_001=0). Evaluaciones suspendidas.")
            
            # Publicar estado IDLE para limpiar el Dashboard en Grafana
            try:
                if hasattr(self, "client") and self.client and self.publisher:
                    # 1. IDLE global
                    self.publisher.publish_metrics(
                        node_id="plc_siemens_lab",
                        device_id="global_plant",
                        health_score_percent=100.0,
                        namur_status="IDLE",
                        mse_raw=0.0,
                        umbral_p95=self.umbral_info.get("p95", 0)
                    )
                    
                    # 2. IDLE individual para cada sensor almacenado
                    for col in self.columnas:
                        topic_real = self.topic_por_columna.get(col)
                        parts = topic_real.split("/") if topic_real else []
                        d_id = parts[-2] if len(parts) >= 2 else "unknown_device"
                        n_id = parts[-3] if len(parts) >= 3 else "unknown_node"
                        
                        p95_sensor = self.umbral_info.get("p95", 0)
                        if self.stats_por_sensor and col in self.stats_por_sensor:
                            p95_sensor = self.stats_por_sensor[col].get("p95", p95_sensor)
                            
                        self.publisher.publish_metrics(
                            node_id=n_id,
                            device_id=d_id,
                            health_score_percent=100.0,
                            namur_status="IDLE",
                            mse_raw=0.0,
                            umbral_p95=p95_sensor
                        )
                    if self.verbose:
                        print(f"    [MQTT] Estado publicado: IDLE (Planta apagada) global y por sensor")
            except Exception as e:
                print(f"    [WARN] Excepción al enviar payload IDLE: {e}")

            self.ultima_eval = ahora
            return

        self.ultima_eval = ahora
        self.n_evaluaciones += 1

        r = self.evaluar_muestra()
        ts = r["ts"]
        es_anomalia = r["es_anomalia"]
        mse_total = r["mse_total"]
        desviacion_ratio = r["desviacion_ratio"]
        errores_features = r["errores_features"]
        reconstruido_raw = r["reconstruido_raw"]
        idx_peor = r["idx_peor"]
        peor_sensor = r["peor_sensor"]
        lista_sensores = r["lista_sensores"]
        salud_planta = r["salud_planta"]
        peor_estado_idx = r["peor_estado_idx"]
        estado_planta = r["estado_planta"]
        n_sensores_degradados = r["n_sensores_degradados"]

        # ── PUBLICAR MÉTRICAS CONTINUAS ──
        try:
            if hasattr(self, "client") and self.client and self.publisher:
                # 1. Publicar estado general de la planta
                self.publisher.publish_metrics(
                    node_id="plc_siemens_lab",
                    device_id="global_plant",
                    health_score_percent=salud_planta,
                    namur_status=estado_planta,
                    mse_raw=mse_total,
                    umbral_p95=self.umbral_info["p95"]
                )

                # 2. Publicar métricas individuales de cada sensor (Root Cause Analysis)
                for sensor in lista_sensores:
                    # Encontrar el umbral p95 específico de este sensor si existe
                    p95_sensor = self.umbral_info["p95"]
                    if self.stats_por_sensor and sensor["tag_name"] in self.stats_por_sensor:
                        p95_sensor = self.stats_por_sensor[sensor["tag_name"]].get("p95", p95_sensor)

                    self.publisher.publish_metrics(
                        node_id=sensor["node_id"],
                        device_id=sensor["device_id"],
                        health_score_percent=sensor["salud_pct"],
                        namur_status=sensor["estado_namur"],
                        mse_raw=sensor["error_reconstruccion"],
                        umbral_p95=p95_sensor
                    )
                
                if self.verbose:
                    print(f"    [MQTT] Predicción publicada: salud={salud_planta:.1f}% estado={estado_planta}")
        except Exception as e:
            print(f"    [WARN] Excepción al enviar payload de métricas: {e}")

        # ── LÓGICA DE HISTÉRESIS / ANTI-FLAPPING (Regla 4.2) ──
        self.historial_anomalias.append(es_anomalia)
        if len(self.historial_anomalias) > self.VENTANA_M:
            self.historial_anomalias.pop(0)

        es_anomalia_confirmada = sum(self.historial_anomalias) >= self.CONFIRMACIONES_N

        if es_anomalia_confirmada and not self.alarma_activa:
            self.n_anomalias += 1
            self.alarma_activa = True

            # Extraer info del sensor crítico para la consola
            topic_real = self.topic_por_columna.get(peor_sensor)
            parts = topic_real.split("/") if topic_real else []
            device_id = parts[-2] if len(parts) >= 2 else peor_sensor
            node_id = parts[-3] if len(parts) >= 3 else "unknown"

            print(f"\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"  !! ALERTA: ANOMALÍA DETECTADA EN SENSADO  #{self.n_anomalias}")
            print(f"  !! Timestamp:        {ts}")
            print(f"  !! MSE Muestra:      {mse_total:.6f} (Umbral: {self.umbral_valor:.6f})")
            print(f"  !! Desviación:       {desviacion_ratio:.1f}x sobre el límite")
            print(f"  !! Sensor Crítico:   {peor_sensor} (Nodo: {node_id} | Dispositivo: {device_id})")
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
                "device_id": device_id,
                "tag_name": peor_sensor,
                "valores": dict(self.buffer),
                "errores": {col: float(errores_features[idx]) for idx, col in enumerate(self.columnas)},
            })
            
            try:
                if hasattr(self, "client") and self.client and self.publisher:
                    sensores_ordenados = sorted(lista_sensores, key=lambda x: x['error_reconstruccion'], reverse=True)
                    root_cause = [{"sensor": s["tag_name"], "deviation_score": s["error_reconstruccion"]} for s in sensores_ordenados[:3]]
                    
                    self.publisher.publish_alert(
                        node_id=node_id,
                        device_id=device_id,
                        health_score_percent=salud_planta,
                        namur_status=estado_planta,
                        mse_raw=mse_total,
                        umbral_p95=self.umbral_info["p95"],
                        root_cause_top=root_cause
                    )
                    print(f"  [MQTT] Alerta extrema enviada a: {MQTT_TOPIC_NOTIFICATIONS}")
            except Exception as e:
                print(f"  [WARN] Excepción al enviar alerta: {e}")

        elif not es_anomalia_confirmada:
            # Si bajamos del umbral, se resetea la alarma
            if self.alarma_activa:
                print(f"\n  [INFO] Planta volvió a estado estable. Alarma desactivada.")
                self.alarma_activa = False
                
            if self.verbose:
                print(f"  [{ts}] Lectura normal - MSE={mse_total:.6f} ({mse_total/self.umbral_valor*100:.0f}% del umbral)")
            else:
                if self.n_evaluaciones % 10 == 0:
                    print(f"  [{ts}] Operación normal | Eval #{self.n_evaluaciones} | Anomalías: {self.n_anomalias}")

    def on_connect(self, client, userdata, flags, *args):
        rc = args[0] if args else 0
        if rc == 0:
            print(f"\n  [OK] Conexión establecida con broker MQTT: {MQTT_BROKER}:{MQTT_PORT}")
            
            # Inicializar el publicador con el cliente
            self.publisher = HealthAssessmentPublisher(client, MQTT_TOPIC_NOTIFICATIONS, MQTT_TOPIC_METRICS)

            print(f"  Suscribiéndose a topic principal: {MQTT_TOPIC_SUB}")
            client.subscribe(MQTT_TOPIC_SUB, qos=1)
            # Publicar estado ONLINE (sobrescribe el LWT OFFLINE)
            topic_status = f"{MQTT_TOPIC_PREFIX}/autoencoder/status"
            client.publish(topic_status,
                json.dumps({"status": "ONLINE", "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")}),
                qos=1, retain=True)
            print(f"\n  [OK] Escuchando lecturas industriales...")
        else:
            print(f"  [ERROR] Fallo de autenticación o conexión MQTT. Código: {rc}")

    def on_disconnect(self, client, userdata, *args):
        # Maneja diferentes firmas según la versión de paho-mqtt
        rc = args[1] if len(args) >= 2 else (args[0] if args else 0)
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

        # Dependiendo de la estructura del topic, el tag (ej. PIT_001) puede estar al final
        # o penúltimo si tiene un sufijo de propiedad (ej. /PIT_001/Pressure)
        partes = topic.split("/")
        sensor_candidato_1 = partes[-1]
        sensor_candidato_2 = partes[-2] if len(partes) > 1 else ""
        
        columna = None
        if sensor_candidato_1 in COLUMNAS_FEATURES:
            columna = sensor_candidato_1
        elif sensor_candidato_2 in COLUMNAS_FEATURES:
            columna = sensor_candidato_2
        else:
            if self.mensajes_recibidos <= 5 and self.verbose:
                print(f"  [DEBUG] Topic ignorado: {topic} (Buscando {sensor_candidato_1} o {sensor_candidato_2})")

        if columna is not None:
            self.buffer[columna] = valor
            self.buffer_timestamps[columna] = time.time()
            self.topic_por_columna[columna] = topic
            if self.verbose:
                print(f"  [LECTURA] {columna} = {valor:.4f}")
            self._evaluar()

    def iniciar(self):
        import platform

        try:
            client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id=MQTT_CLIENT_ID
            )
        except AttributeError:
            client = mqtt.Client(client_id=MQTT_CLIENT_ID)

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
