# -*- coding: utf-8 -*-
"""
SAT Camera Frame Capture Service — Entrypoint & REST API.
Microservicio para captura de fotogramas de cámara industrial.
"""

import sys
import os
import argparse
import logging

try:
    from fastapi import FastAPI, Response, HTTPException
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    FastAPI = None

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

try:
    from config import config
    from camera_capturer import CameraCapturer
except ImportError:
    from .config import config
    from .camera_capturer import CameraCapturer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("SATCameraMain")

capturer = CameraCapturer()

if HAS_FASTAPI:
    app = FastAPI(
        title="SAT Camera Frame Capture Service",
        description="Microservicio REST para captura de fotogramas de camara industrial",
        version="1.0.0"
    )

    @app.get("/health")
    def health_check():
        return {
            "status": "ONLINE",
            "camera_url": config.CAMERA_URL,
            "timeout_seconds": config.CAMERA_TIMEOUT_SECONDS
        }

    @app.get("/capture")
    def capture_frame():
        frame_bytes = capturer.capturar_frame()
        if not frame_bytes:
            raise HTTPException(status_code=503, detail="No se pudo capturar fotograma de la camara")
        return Response(content=frame_bytes, media_type="image/jpeg")


def main():
    parser = argparse.ArgumentParser(description="SAT Camera Frame Capture Microservice")
    parser.add_argument("--snapshot", action="store_true", help="Capturar fotograma y guardar en disco")
    parser.add_argument("--output", type=str, default="captura_camara.jpg", help="Ruta del archivo de salida")
    parser.add_argument("--serve", action="store_true", help="Iniciar servidor HTTP REST API")
    parser.add_argument("--port", type=int, default=config.PORT, help="Puerto del servidor REST API")
    args = parser.parse_args()

    if args.snapshot:
        logger.info(f"Capturando fotograma desde {config.CAMERA_URL}...")
        img_bytes = capturer.capturar_frame()
        if not img_bytes:
            logger.error("No se pudo capturar fotograma de la camara.")
            sys.exit(1)

        with open(args.output, "wb") as f:
            f.write(img_bytes)
        logger.info(f"Fotograma guardado en: {os.path.abspath(args.output)}")
        return

    if args.serve or not sys.argv[1:]:
        if not HAS_FASTAPI:
            logger.error("FastAPI/Uvicorn no instalados. Ejecute: pip install fastapi uvicorn")
            sys.exit(1)
        logger.info(f"Iniciando SAT Camera REST API en puerto {args.port}...")
        uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
