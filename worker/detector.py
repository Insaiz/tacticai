from __future__ import annotations
import os
from pathlib import Path

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None


class FootballDetector:
    def __init__(self):
        self.model = None
        if YOLO is None:
            return

        path = os.getenv("TACTICAI_YOLO_MODEL", "").strip()
        if not path:
            cache = Path(os.getenv("TACTICAI_MODEL_CACHE", "/tmp/tacticai-models"))
            cache.mkdir(parents=True, exist_ok=True)
            repo = os.getenv("TACTICAI_YOLO_REPO", "hugosanchezgallego/futbol-detector-v4")
            filename = os.getenv("TACTICAI_YOLO_FILENAME", "best.pt")
            path = str(cache / filename)
            if not Path(path).exists() and hf_hub_download is not None:
                path = hf_hub_download(
                    repo_id=repo,
                    filename=filename,
                    local_dir=str(cache),
                    local_dir_use_symlinks=False,
                )

        if path and Path(path).exists():
            self.model = YOLO(path)

    @property
    def available(self):
        return self.model is not None

    def detect(self, frame):
        if not self.model:
            return []

        r = self.model.predict(
            frame,
            conf=float(os.getenv("TACTICAI_DETECTOR_CONF", "0.15")),
            imgsz=int(os.getenv("TACTICAI_DETECTOR_IMGSZ", "1280")),
            verbose=False,
        )[0]

        out = []
        for b in r.boxes:
            cls = int(b.cls[0])
            conf = float(b.conf[0])
            x1, y1, x2, y2 = map(float, b.xyxy[0])
            out.append({
                "class_id": cls,
                "confidence": conf,
                "x": (x1 + x2) / 2,
                "y": (y1 + y2) / 2,
                "w": x2 - x1,
                "h": y2 - y1,
            })
        return out
