import os
import json
import time
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

# Load .env
load_dotenv(".env")

MQTT_BROKER = os.getenv("MQTT_BROKER", "192.168.10.208")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_USER = os.getenv("MQTT_USER", "sat_lab")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "&HjVFmrhuBK")

# Base sensors list (16 variables)
sensores = [
    ("PIT_001", 10.5),
    ("FIT_001_MAS", 50.0),
    ("FIT_001_DENS", 998.0),
    ("FIT_001_TEMP", 25.0),
    ("FIT_001_VOL", 50.1),
    ("LIT_001", 80.0),
    ("LIT_002", 75.0),
    ("TT_001", 24.5),
    ("Bomba_Agua_001_STATUS", 1.0),
    ("Bomba_Agua_001_REF", 1500.0),
    ("Val_001", 1.0),
    ("Val_002", 0.0),
    ("Val_003", 1.0),
    ("Val_004", 0.0),
    ("Mot_Comp_001", 1.0),
    ("MOTOR_01", 1.0),
]

def simulate_data():
    client = mqtt.Client(client_id="simulador_test")
    client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
    client.connect(MQTT_BROKER, MQTT_PORT)
    
    print(f"Conectado a MQTT {MQTT_BROKER}:{MQTT_PORT} simulando datos...")
    
    node_id = "nodo_principal"
    device_id = "plc_01"
    
    for tag_name, val_base in sensores:
        topic = f"sat_lab/telemetry/{node_id}/{device_id}/{tag_name}"
        payload = json.dumps({
            "value": val_base,
            "timestamp": int(time.time() * 1000)
        })
        client.publish(topic, payload, qos=1)
        print(f"Publicado {tag_name}: {val_base}")
        time.sleep(0.1) # Simulate slight delay between sensor reads
        
    print("Muestra completa publicada. Esperando que el modelo evalue (intervalo 15s)...")
    
    # Send another one to trigger anomalies or normal behavior
    time.sleep(16)
    print("\nEnviando ráfaga de 5 muestras anómalas para superar el debounce...")
    for i in range(5):
        for tag_name, val_base in sensores:
            if tag_name == "PIT_001":
                val = 900.0 + i # Anomaly!
            elif tag_name == "TT_001":
                val = 150.0 + i # Anomaly!
            else:
                val = val_base
                
            topic = f"sat_lab/telemetry/{node_id}/{device_id}/{tag_name}"
            payload = json.dumps({
                "value": val,
                "timestamp": int(time.time() * 1000)
            })
            client.publish(topic, payload, qos=1)
        time.sleep(1)
        
    print("Muestras anómalas publicadas. Desconectando...")
    client.disconnect()

if __name__ == "__main__":
    simulate_data()
