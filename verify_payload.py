# -*- coding: utf-8 -*-
"""
Script de verificación offline para validar el formato del JSON de anomalías.
"""
import sys
import json
import numpy as np
from detectar_mqtt import DetectorAnomaliasMQTT

class MockMQTTClient:
    def __init__(self):
        self.published = []

    def publish(self, topic, payload, **kwargs):
        self.published.append((topic, payload))
        print(f"\n[MOCK MQTT] Publicando en '{topic}' (qos={kwargs.get('qos', 0)}, retain={kwargs.get('retain', False)}):")
        try:
            data = json.loads(payload)
            print(json.dumps(data, indent=2, ensure_ascii=False))
        except (json.JSONDecodeError, TypeError):
            print(payload)

CAMPOS_REQUERIDOS_SENSOR = [
    "equipo_id", "metrica", "unidad", "tipo_dato",
    "valor_actual", "valor_esperado", "error_absoluto",
    "error_reconstruccion", "salud_pct", "estado_namur",
    "estado_code", "es_sensor_anomalo", "es_critico",
    "quality", "timestamp_dato",
]

CAMPOS_REQUERIDOS_ROOT = ["schema_version", "timestamp", "modelo", "planta", "sensores"]

def verificar():
    print("Iniciando prueba offline del Detector...")
    
    detector = DetectorAnomaliasMQTT(umbral_tipo="p95", verbose=True, intervalo_eval=0)
    detector.client = MockMQTTClient()
    
    for col in detector.columnas:
        detector.buffer[col] = 10.0
        
    detector.buffer["LIT_002_Level"] = 150.0
    
    print("\nEjecutando _evaluar()...")
    detector._evaluar()
    
    # Validar lo publicado
    if not detector.client.published:
        print("\n[ERROR] No se publicó ningún payload.")
        return

    topic, raw = detector.client.published[0]
    data = json.loads(raw)

    print("\n" + "=" * 60)
    print("  VALIDACIÓN DE CAMPOS DEL PAYLOAD")
    print("=" * 60)

    errores = 0
    # Root
    for campo in CAMPOS_REQUERIDOS_ROOT:
        if campo not in data:
            print(f"  [FALTA] Root: '{campo}'")
            errores += 1
        else:
            print(f"  [OK]    Root: '{campo}' = {json.dumps(data[campo], ensure_ascii=False)[:80]}")

    # Sensores
    for i, sensor in enumerate(data.get("sensores", [])):
        for campo in CAMPOS_REQUERIDOS_SENSOR:
            if campo not in sensor:
                print(f"  [FALTA] Sensor {i} ({sensor.get('equipo_id','?')}): '{campo}'")
                errores += 1

    if errores == 0:
        print(f"\n  ✅ Payload validado correctamente — {len(CAMPOS_REQUERIDOS_ROOT)} campos root + {len(CAMPOS_REQUERIDOS_SENSOR)} campos/sensor")
    else:
        print(f"\n  ❌ {errores} campos faltantes")

if __name__ == "__main__":
    verificar()

