# -*- coding: utf-8 -*-
"""
Configuración centralizada para SAT Camera Service.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()


@dataclass(frozen=True)
class CameraConfig:
    CAMERA_URL: str = os.getenv("CAMERA_URL", "http://192.168.10.44:8889/laboratorio-cam/")
    CAMERA_SNAPSHOT_URL: str = os.getenv("CAMERA_SNAPSHOT_URL", "http://192.168.10.44:8889/laboratorio-cam/frame.jpeg")
    CAMERA_RTSP_URL: str = os.getenv("CAMERA_RTSP_URL", "rtsp://192.168.10.44:8554/laboratorio-cam")
    CAMERA_TIMEOUT_SECONDS: int = int(os.getenv("CAMERA_TIMEOUT_SECONDS", 4))
    CAMERA_ENABLED: bool = os.getenv("CAMERA_ENABLED", "true").lower() in ("true", "1", "yes")
    PORT: int = int(os.getenv("PORT", 8005))


config = CameraConfig()
