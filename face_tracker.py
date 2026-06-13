import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "assets", "face_landmarker.task")

# Roda na mesma resolução reduzida das mãos — landmarks são normalizados (0-1),
# então remapeiam corretamente para o frame completo.
PROC_W, PROC_H = 640, 360


def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Baixando modelo de rosto (~4 MB)...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Modelo de rosto pronto.")


class FaceTracker:
    """Mapeia o rosto via MediaPipe Tasks (FaceLandmarker) — mesma API das mãos."""

    def __init__(self, max_faces=1):
        self._ok      = False
        self._result  = None
        try:
            _ensure_model()
            BaseOptions    = mp.tasks.BaseOptions
            FaceLandmarker = mp.tasks.vision.FaceLandmarker
            Options        = mp.tasks.vision.FaceLandmarkerOptions
            RunningMode    = mp.tasks.vision.RunningMode

            opts = Options(
                base_options=BaseOptions(model_asset_path=MODEL_PATH),
                running_mode=RunningMode.VIDEO,
                num_faces=max_faces,
            )
            self._lm    = FaceLandmarker.create_from_options(opts)
            self._t0    = time.monotonic()
            self._small = np.empty((PROC_H, PROC_W, 3), dtype=np.uint8)
            self._ok    = True
            print("FaceTracker: modelo carregado.")
        except Exception as e:
            print("AVISO: FaceTracker indisponível —", e)

    def process(self, frame: np.ndarray):
        if not self._ok:
            return
        cv2.resize(frame, (PROC_W, PROC_H), dst=self._small)
        rgb    = cv2.cvtColor(self._small, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms  = int((time.monotonic() - self._t0) * 1000)
        try:
            self._result = self._lm.detect_for_video(mp_img, ts_ms)
        except Exception:
            self._result = None

    def get_face(self, frame: np.ndarray):
        """Retorna dict {idx: (x, y)} no frame completo, ou None."""
        if not self._ok or not self._result or not self._result.face_landmarks:
            return None
        h, w    = frame.shape[:2]
        lm_list = self._result.face_landmarks[0]
        return {i: (int(p.x * w), int(p.y * h)) for i, p in enumerate(lm_list)}

    def release(self):
        if self._ok:
            try:
                self._lm.close()
            except Exception:
                pass
