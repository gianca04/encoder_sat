import json
import time
from typing import List, Dict, Any

class HealthAssessmentPublisher:
    """
    Clase encargada de estructurar y publicar diagnósticos de salud 
    (Health Assessment) a través de MQTT, siguiendo estándares como ISO 13374
    e integrando el control de ciclo de vida (BIRTH/DEATH) de Sparkplug B.
    """
    
    def __init__(self, mqtt_client, topic_notifications: str, topic_metrics: str):
        """
        Inicializa el publicador de salud.
        
        Args:
            mqtt_client: Instancia conectada de paho.mqtt.client
            topic_notifications (str): Topic para alertas extremas.
            topic_metrics (str): Topic para publicación de predicciones continuas.
        """
        self.client = mqtt_client
        self.topic_notifications = topic_notifications
        self.topic_metrics = topic_metrics

    @staticmethod
    def setup_sparkplug_lwt(mqtt_client, node_id: str, group_id: str = "Grupo_SAT") -> None:
        """
        Configura el Testamento (Last Will and Testament - NDEATH) en el cliente MQTT.
        ¡ADVERTENCIA!: Debe llamarse ANTES de ejecutar mqtt_client.connect().
        """
        topic_ndeath = f"spBv1.0/{group_id}/NDEATH/{node_id}"
        payload = {
            "timestamp": int(time.time() * 1000),
            "status": "OFFLINE"
        }
        mqtt_client.will_set(topic=topic_ndeath, payload=json.dumps(payload), qos=1, retain=False)

    def publish_nbirth(self, node_id: str, group_id: str = "Grupo_SAT") -> bool:
        """Avisa a la red que este script (Nodo) acaba de arrancar y conectarse."""
        if self.client is None:
            return False
            
        topic = f"spBv1.0/{group_id}/NBIRTH/{node_id}"
        payload = {
            "timestamp": int(time.time() * 1000),
            "status": "ONLINE",
            "metrics": [{"name": "node_status", "type": "String", "value": "RUNNING"}]
        }
        self.client.publish(topic, json.dumps(payload), qos=1)
        return True

    def publish_dbirth(self, node_id: str, device_id: str, group_id: str = "Grupo_SAT") -> bool:
        """Avisa a la red que el modelo va a empezar a mandar predicciones para un dispositivo."""
        if self.client is None:
            return False
            
        topic = f"spBv1.0/{group_id}/DBIRTH/{node_id}/{device_id}"
        payload = {
            "timestamp": int(time.time() * 1000),
            "status": "ONLINE",
            "metrics": [
                {"name": "health_score_percent", "type": "Float"},
                {"name": "namur_status", "type": "String"},
                {"name": "mse_raw", "type": "Float"},
                {"name": "threshold_p95", "type": "Float"}
            ]
        }
        self.client.publish(topic, json.dumps(payload), qos=1)
        return True

    def publish_ddeath(self, node_id: str, device_id: str, group_id: str = "Grupo_SAT") -> bool:
        """Avisa a la red que las predicciones para este dispositivo se han pausado o detenido."""
        if self.client is None:
            return False
            
        topic = f"spBv1.0/{group_id}/DDEATH/{node_id}/{device_id}"
        payload = {
            "timestamp": int(time.time() * 1000),
            "status": "OFFLINE"
        }
        self.client.publish(topic, json.dumps(payload), qos=1)
        return True

    def publish_metrics(
        self, 
        node_id: str,
        device_id: str,
        health_score_percent: float, 
        namur_status: str, 
        mse_raw: float,
        umbral_p95: float
    ) -> bool:
        """
        Publica las predicciones y métricas del modelo de forma continua.
        """
        if self.client is None:
            return False

        payload = {
            "timestamp": int(time.time() * 1000),
            "asset": {
                "node_id": node_id,
                "device_id": device_id
            },
            "metrics": {
                "health_score_percent": round(health_score_percent, 2),
                "namur_status": namur_status,
                "mse_raw": round(mse_raw, 4),
                "threshold_p95": round(umbral_p95, 4)
            }
        }
        
        self.client.publish(self.topic_metrics, json.dumps(payload), qos=0)
        return True

    def publish_alert(
        self, 
        node_id: str,
        device_id: str,
        health_score_percent: float, 
        namur_status: str, 
        mse_raw: float,
        umbral_p95: float,
        root_cause_top: List[Dict[str, Any]],
        alert_msg: str = "CRITICAL ANOMALY DETECTED"
    ) -> bool:
        """
        Estructura el payload y lo publica solo cuando hay una alerta extrema.
        """
        if self.client is None:
            return False

        payload = {
            "alert": alert_msg,
            "timestamp": int(time.time() * 1000),
            "asset": {
                "node_id": node_id,
                "device_id": device_id
            },
            "health_assessment": {
                "health_score_percent": round(health_score_percent, 2),
                "namur_status": namur_status,
                "deviation_pct": round((mse_raw / umbral_p95) * 100, 2) if umbral_p95 > 0 else 0
            },
            "root_cause_analysis": root_cause_top
        }
        
        self.client.publish(self.topic_notifications, json.dumps(payload), qos=1)
        return True
