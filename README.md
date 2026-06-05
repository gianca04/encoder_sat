# Autoencoder Anomaly Detection Pipeline 🚀

Este repositorio contiene un sistema completo, minimalista y de nivel de producción para la **detección de anomalías en tiempo real** en 6 sensores industriales utilizando redes neuronales **Autoencoder (Keras)** y comunicación **MQTT / Sparkplug B via EMQX**.

---

## 📂 Estructura del Repositorio

* **`entrenar.ipynb`**: Jupyter Notebook interactivo para realizar la extracción, preprocesamiento y entrenamiento de forma visual e interactiva con gráficas inline.
* **`entrenar.py`**: Script unificado optimizado para la ejecución automática en segundo plano (daemon o cron job).
* **`detectar_mqtt.py`**: Script de detección en tiempo real que se conecta a EMQX, analiza los datos de los sensores y publica alertas y estados en vivo.
* **`utils.py`**: Configuración de sensores, rangos normales y utilidades compartidas.
* **`requirements.txt`**: Librerías y dependencias necesarias.

---

## 🛠️ Instalación de Dependencias

Antes de comenzar, instala las dependencias necesarias ejecutando:

```bash
pip install -r requirements.txt
```

---

## 📈 1. Entrenamiento del Modelo

El modelo se entrena de forma segura utilizando datos históricos reales de **VictoriaMetrics** desde el inicio del proyecto (**29/05/2026 00:00:00**). Antes de sobrescribir, el sistema realiza un backup automático en `modelos/backup/` por seguridad.

### Opción A: Entrenamiento Visual e Interactivo (Notebook)
Para experimentar y ver las gráficas de pérdida (loss) e histogramas de error MSE:
```bash
jupyter notebook entrenar.ipynb
```

### Opción B: Entrenamiento Automático por Daemon (Script CLI)
Para reentrenar de forma desatendida y periódica desde el terminal:
```bash
# Entrenamiento estándar (desde el inicio del proyecto hasta ahora)
python entrenar.py --start-date "2026-05-29T00:00:00"

# Opciones avanzadas (especificando epochs o rango cerrado de fechas)
python entrenar.py --start-date "2026-05-29T00:00:00" --epochs 100 --batch-size 64
```

---

## 📡 2. Detección en Tiempo Real (MQTT)

El detector se conecta a tu broker **EMQX**, recibe las métricas decodificadas y publica alertas instantáneas.

### Ejecutar el Detector:
```bash
# Modo normal (muestra resúmenes cada 10 evaluaciones)
python detectar_mqtt.py

# Modo depuración (muestra cada lectura recibida en tiempo real)
python detectar_mqtt.py --verbose

# Configurar umbral estricto (P99 en lugar de P95) e intervalo de evaluación
python detectar_mqtt.py --umbral p99 --intervalo 10
```

### 🔔 Canales de Comunicación y Alertas en EMQX:
* **Entrada de Datos (Subscripción):** El script lee las métricas planas publicadas en `lab_sat/<equipo>/<metrica>`.
* **Publicación de Estados Continuos (`0` o `1`):** Publica de forma continua `0` (Normal) o `1` (Anómalo) en `lab_sat/<equipo>/<metrica>/anomalia` para activar alarmas visuales en SCADAs o Grafana.
* **Publicación de Reporte JSON en Alertas:** Si se detecta una anomalía real, publica un JSON detallado en **`lab_sat/anomalias`** incluyendo el MSE de desviación y los valores actuales vs esperados:
  ```json
  {
    "timestamp": "2026-06-02 19:53:00 UTC",
    "n_anomalia": 1,
    "mse_total": 0.045123,
    "umbral_valor": 0.015000,
    "umbral_tipo": "p95",
    "sensor_critico": "FIT_001_Flow",
    "valor_actual": 12.345,
    "valor_esperado": 3.456,
    "desviacion_ratio": 3.008
  }
  ```
  *   **`timestamp`**: Marca de tiempo UTC en que se evaluó y detectó la anomalía.
  *   **`n_anomalia`**: Contador secuencial de las anomalías detectadas durante la sesión activa.
  *   **`mse_total`**: Error de reconstrucción global (Mean Squared Error) de la muestra multivariada actual.
  *   **`umbral_valor`**: Valor numérico límite del umbral cargado del Autoencoder.
  *   **`umbral_tipo`**: Tipo de umbral utilizado (`p95` o `p99`).
  *   **`sensor_critico`**: Identificador del sensor con la mayor contribución al error de reconstrucción (sensor causante de la alerta).
  *   **`valor_actual`**: Lectura en tiempo real recibida del sensor crítico.
  *   **`valor_esperado`**: Valor normal de referencia estimado por el Autoencoder para ese sensor específico.
  *   **`desviacion_ratio`**: Factor de gravedad de la anomalía (`mse_total / umbral_valor`). Por ejemplo, un valor de `3.0` indica que el error es 3 veces el límite permitido.

* **Publicación de Métricas de Monitoreo Continuo (JSON):** Publica en cada ciclo de evaluación en el topic **`lab_sat/autoencoder/metricas`** un reporte estructurado de salud global de planta e individual por sensor:
  ```json
  {
    "schema_version": "2.0",
    "timestamp": "2026-06-03T14:49:24+00:00",
    "modelo": {
      "id": "autoencoder_v3",
      "umbral_tipo": "p95",
      "umbral_valor": 0.000183
    },
    "planta": {
      "salud_pct": 0.0,
      "estado_namur": "CRITICAL",
      "estado_code": 3,
      "sensores_degradados": 2,
      "sensores_total": 6,
      "mse": 0.000281,
      "desviacion_ratio": 1.54
    },
    "sensores": [
      {
        "equipo_id": "LIT_002",
        "metrica": "Level",
        "unidad": "m",
        "tipo_dato": "REAL",
        "valor_actual": 56.9951,
        "valor_esperado": 57.1912,
        "error_absoluto": 0.1961,
        "error_reconstruccion_norm": 0.001366,
        "salud_pct": 0.0,
        "estado_namur": "CRITICAL",
        "estado_code": 3,
        "es_sensor_anomalo": true,
        "es_critico": true,
        "quality": "GOOD",
        "timestamp_dato": "2026-06-03T14:49:22+00:00"
      }
    ]
  }
  ```

---

## 📋 Norma de Diagnóstico de Salud (ISO 13374 / ISO 17359)

El sistema evalúa el error de reconstrucción (MSE) individual de cada sensor para calcular su porcentaje de salud (`salud_pct`) e inferir su estado funcional. Para mantener la compatibilidad con el esquema original del JSON (etiquetas `estado_namur`), se inyectan directamente las nomenclaturas de la norma de monitoreo de condición de máquinas **ISO 13374** (Asset Health Index - AHI):

### Estados y Rangos del Índice de Salud (AHI)

| Rango de Salud (%) | Nomenclatura ISO (`estado_namur`) | Código (`estado_code`) | Color SCADA / Grafana | Descripción Operacional |
| :---: | :--- | :---: | :--- | :--- |
| **85% - 100%** | **`OPTIMAL`** | **`0`** | 🟩 **Verde** | **Normal:** Operación óptima. Sin anomalías (error $\le P_{95}$). |
| **70% - 84.9%** | **`ACCEPTABLE`** | **`1`** | 🟨 **Amarillo** | **Alerta Temprana:** Deriva inicial leve detectada. |
| **50% - 69.9%** | **`DEGRADED`** | **`2`** | 🟧 **Naranja** | **Degradado:** Desviación clara de patrones normales. Requiere planificar intervención. |
| **0% - 49.9%** | **`CRITICAL`** | **`3`** | 🟥 **Rojo** | **Falla Crítica:** Riesgo inminente de avería. Intervención correctiva urgente (error $\ge$ umbral crítico). |

### Lógica de Agregación de Planta (Principio del Eslabón más Débil)

El estado general de la planta (`planta.estado_namur` y `planta.estado_code`) se determina de forma robusta bajo el principio industrial del eslabón más débil (*weakest link*):
1. **Salud General (`salud_pct`):** Se reporta como el valor mínimo de salud observado entre todos los sensores monitoreados.
2. **Estado General (`estado_code` / `estado_namur`):** Se asigna el peor estado (código numérico más alto) de todos los sensores individuales. Si un solo sensor crítico entra en `CRITICAL`, toda la planta pasa automáticamente a estado `CRITICAL`.



