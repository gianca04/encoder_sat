# -*- coding: utf-8 -*-
"""
Capturador de Fotogramas de Cámara Industrial.
Captura fotogramas binarios JPEG desde la cámara vía HTTP Snapshot o flujo RTSP con OpenCV.
"""

import logging
from typing import Optional
import requests

try:
    from config import config
except ImportError:
    from .config import config

logger = logging.getLogger("SATCameraCapturer")


class CameraCapturer:
    def __init__(self, timeout: Optional[int] = None):
        self.timeout = timeout or config.CAMERA_TIMEOUT_SECONDS

    def capturar_frame(self) -> Optional[bytes]:
        """Obtiene la imagen binaria JPEG desde la cámara."""
        if not config.CAMERA_ENABLED:
            return None

        # 1. Intentar HTTP Snapshot
        url_snap = config.CAMERA_SNAPSHOT_URL or config.CAMERA_URL
        try:
            resp = requests.get(url_snap, timeout=self.timeout)
            if resp.status_code == 200 and resp.content:
                if resp.content.startswith(b"\xff\xd8") or resp.content.startswith(b"\x89PNG"):
                    logger.info("Fotograma capturado vía HTTP Snapshot.")
                    return resp.content
        except Exception as e:
            logger.debug(f"HTTP Snapshot falló: {e}")

        # 2. Intentar flujo RTSP con OpenCV (MediaMTX puerto 8554)
        url_rtsp = config.CAMERA_RTSP_URL or "rtsp://192.168.10.44:8554/laboratorio-cam"
        try:
            import cv2
            cap = cv2.VideoCapture(url_rtsp)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    success, encoded_img = cv2.imencode(".jpg", frame)
                    if success:
                        logger.info("Fotograma capturado vía OpenCV RTSP Stream.")
                        return encoded_img.tobytes()
        except Exception as e:
            logger.debug(f"OpenCV RTSP falló: {e}")

        logger.warning(f"No se pudo obtener imagen binaria de la cámara.")
        return None
