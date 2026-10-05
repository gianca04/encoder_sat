---
trigger: always_on
---

# RULES: Desarrollo de Autoencoders para Detección y Mantenimiento Predictivo (IIoT / Industria 4.0)

## 1. Integridad de Datos OT y Gestión de Regímenes Operativos

1.1. FILTRADO DE ESTADOS DE MÁQUINA (Golden Baseline):

- El agente NUNCA debe entrenar un Autoencoder con datos mezclados de máquina apagada, paradas de planta o estados transitorios no estacionarios, a menos que el modelo sea explícitamente condicional o multi-modo.
- DEBE aplicar una máscara o filtro basado en tags de estado operativo (ej. `MOTOR_RUNNING == 1`, `SPEED > nominal * 0.1` o flags de PLC/SCADA) antes de calcular normalizaciones y entrenar.

1.2. MANEJO DE GAPS TEMPORALES Y CALIDAD DE SEÑAL:

- En telemetría industrial (MQTT, OPC-UA, historians), los vacíos de datos (desconexiones, paradas de fin de semana) NO DEBEN ser interpolados de forma continua a ciegas.
- DEBE segmentar las series en sub-bloques temporales continuos (ej. si `delta_time > max_gap_segundos`, cortar el bloque) e interpolar únicamente DENTRO del bloque válido.
- DEBE descartar o imputar con flags datos con *Quality Code* inválido (Bad/Uncertain) o señales estancadas (*stale data* con varianza cero durante periodos anómalos).

1.3. PREVENCIÓN DE FUGA DE INFORMACIÓN (Data Leakage):

- El ajuste del scaler (`fit`) DEBE realizarse EXCLUSIVAMENTE sobre el subconjunto cronológico de entrenamiento libre de fallas conocidas (`X_train`).
- NUNCA aplicar `fit_transform` sobre el dataset completo ni utilizar división aleatoria (`train_test_split(shuffle=True)`). En series de tiempo industriales, la partición DEBE ser estrictamente temporal/cronológica.

---

## 2. Arquitectura de Red y Dinámica del Modelo

2.1. CONSISTENCIA ENTRE ESCALADO Y FUNCIÓN DE SALIDA:

- Si el preprocesamiento utiliza `MinMaxScaler(feature_range=(0, 1))`, la capa de salida de reconstrucción DEBE utilizar activación `sigmoid` o acotada.
- Si se utiliza `StandardScaler` o `RobustScaler` (valores centrados en 0 con colas negativas y positivas), la capa de salida DEBE ser estrictamente lineal (`activation=None` o `linear`).
- Para señales con picos y ruido de proceso, se PREFIERE `RobustScaler` o `QuantileTransformer` frente a `MinMaxScaler` cuando existan outliers de instrumentación.

2.2. TOPOLOGÍA SEGÚN LA DINÁMICA DEL SISTEMA:

- **Autoencoders Densos (MLP)**: Permitidos ÚNICAMENTE cuando las variables presentan relaciones inter-sensor estáticas o casi instantáneas en el mismo instante $t$.
- **Autoencoders Recurrentes o Convolucionales (LSTM, GRU, TCN, 1D-CNN)**: REQUERIDOS cuando la anomalía dependa de la secuencia temporal, inercia térmica, rampas de aceleración o memoria de proceso ($t-W$ hasta $t$).

2.3. CONTROL DEL CUELLO DE BOTELLA (Bottleneck):

- La dimensión latente DEBE representar una reducción significativa respecto a las dimensiones de entrada (típicamente $\le 50\%$ de los features o grados de libertad físicos del proceso), para forzar al modelo a aprender las relaciones de conservación de masa/energía del sistema y evitar la reconstrucción trivial o sobreajuste al ruido electromagnético.

---

## 3. Umbrales de Anomalía y Diagnóstico de Salud (Health Assessment)

3.1. ESTIMACIÓN ESTADÍSTICA DE UMBRALES (Thresholding):

- NO usar umbrales "hardcodeados" arbitrarios. El umbral base de corte DEBE calcularse sobre los residuos de reconstrucción de datos normales de validación.
- Métodos exigidos:
  - Percentiles empíricos no paramétricos ($P_{95}$ para Advertencia/Warning, $P_{99}$ para Alarma Crítica).
  - Opcional avanzado: Umbrales adaptativos basados en Teoría de Valores Extremos (EVT / POT: Peak Over Threshold) o distancia de Mahalanobis sobre el vector de residuos si los sensores están fuertemente correlacionados.

3.2. CÁLCULO DE SALUD POR SENSOR Y EXPLICABILIDAD (Root Cause):

- El modelo NO DEBE limitarse a emitir un escalar booleano global (`is_anomaly: True/False`).
- DEBE calcular y registrar el error cuadrático individual por feature o sensor:
    $$e_i(t) = (x_i(t) - \hat{x}_i(t))^2$$
- DEBE identificar y reportar cuál es el sensor o subsistema con mayor contribución porcentual al error global de reconstrucción.

3.3. ESTÁNDARES INDUSTRIALES:

- La salida del sistema de detección DEBE mapearse o ser compatible con la arquitectura de monitoreo de condición **ISO 13374** (Data Acquisition $\to$ Data Manipulation $\to$ State Detection $\to$ Health Assessment).
- La clasificación de severidad DEBE respetar categorías de estado de instrumentos tipo **NAMUR NE 107**:
  - Normal / Optimal
  - Maintenance Required (Degradación leve / P95)
  - Out of Specification (P98)
  - Failure / Critical (P99+)

---

## 4. Inferencia en Streaming y Operación IIoT (MQTT / Kafka / Edge)

4.1. BUFFER MULTIVARIADO Y ASINCRONISMO:

- En protocolos asíncronos orientados a mensajes (ej. MQTT), los sensores publican a distintas frecuencias y marcas de tiempo.
- El código de inferencia DEBE implementar un buffer multivariado sincronizado con verificación de frescura (*TTL / Stale Timeout*). Si un sensor excede el timeout de actualización, la muestra NO se evalúa o se marca como señal inválida para no inferir con valores fantasma.

4.2. HISTÉRESIS Y ANTI-FLAPPING (Debounce de Alarmas):

- Para evitar la fatiga de alarmas generada por picos espurios o transitorios hidráulicos/eléctricos, el sistema de detección NO DEBE disparar alertas críticas con una única muestra por encima del umbral.
- DEBE implementar persistencia temporal: ventana deslizante de confirmación (ej. $N$ de las últimas $M$ evaluaciones por encima del umbral) o suavizado exponencial del residuo (EWMA).

---

## 5. MLOps Industrial: Versionado, Atomicidad y Fail-Safe

5.1. EMPAQUETADO ATÓMICO DE ARTEFACTOS:

- Un modelo nunca es solo la red neuronal. Todo despliegue DEBE guardar y versionar de forma acoplada y atómica:
    1. Pesos y arquitectura del modelo (`.keras` o `.onnx`).
    2. Scaler ajustado (`scaler.joblib`).
    3. Diccionario de umbrales globales y por sensor (`umbral.joblib`).
    4. Metadatos de entrenamiento (lista ordenada exacta de tags/features, unidades, rangos de fecha de baseline, versión del pipeline).

5.2. ESTRATEGIAS DE RESILENCIA (Backup & Rollback):

- Antes de sobreescribir artefactos en producción durante un reentrenamiento diario/semanal, el script DEBE realizar un respaldo (*backup*) del modelo anterior.
- Si el proceso de reentrenamiento, validación o guardado genera una excepción o un *val_loss* inaceptablemente alto, DEBE ejecutarse un rollback automático para no interrumpir el servicio de inferencia en tiempo real.
