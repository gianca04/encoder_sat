# SAT Camera Frame Capture Microservice

Microservicio modular en Python encargado de capturar fotogramas en tiempo real desde la cámara del laboratorio:
`http://192.168.10.44:8889/laboratorio-cam/`

---

## 🛠️ Características

1. **Múltiples Estrategias de Captura**:
   - HTTP GET directo a endpoints de fotogramas (`/frame.jpeg`).
   - Endpoint de API go2rtc / MediaMTX (`http://192.168.10.44:1984/api/frame.jpeg?src=laboratorio-cam`).
   - OpenCV `cv2.VideoCapture` para streams RTSP / MJPEG.
   - Generación de imagen sintética de prueba para desarrollo/fallback cuando la cámara física está desconectada.
2. **REST API Integrada (FastAPI)**:
   - `GET /capture`: Retorna los bytes JPEG del fotograma actual.
   - `GET /health`: Estado del servicio de cámara.
3. **Integración con Telegram**: Permite adjuntar imágenes reales de la planta directamente en las alertas de anomalías del bot de Telegram.

---

## ⚙️ Configuración (`.env`)

```env
CAMERA_URL=http://192.168.10.44:8889/laboratorio-cam/
CAMERA_SNAPSHOT_URL=http://192.168.10.44:8889/laboratorio-cam/frame.jpeg
CAMERA_RTSP_URL=rtsp://192.168.10.44:8554/laboratorio-cam
CAMERA_TIMEOUT_SECONDS=4
CAMERA_ENABLED=true
PORT=8005
```

---

## 🚀 Instalación y Uso

### 1. Instalación de Dependencias
```bash
pip install -r requirements.txt
```

### 2. Capturar Fotograma desde CLI
```bash
python main.py --snapshot --output captura.jpg
```

### 3. Iniciar Servidor REST API
```bash
python main.py --serve --port 8005
```
