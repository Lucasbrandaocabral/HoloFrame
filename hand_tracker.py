import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "assets", "hand_landmarker.task")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17), (5, 17),
]

# MediaPipe roda nessa resolução — landmarks são normalizados (0-1)
# então as coordenadas ainda mapeiam corretamente para o frame completo.
PROC_W, PROC_H = 640, 360


def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Baixando modelo de mãos (~25 MB)...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Modelo pronto.")


class HandTracker:
    THUMB_TIP = 4
    INDEX_TIP = 8

    def __init__(self, max_hands=2, detection_conf=0.7, tracking_conf=0.7):
        _ensure_model()

        BaseOptions        = mp.tasks.BaseOptions
        HandLandmarker     = mp.tasks.vision.HandLandmarker
        HandLandmarkerOpts = mp.tasks.vision.HandLandmarkerOptions
        RunningMode        = mp.tasks.vision.RunningMode

        opts = HandLandmarkerOpts(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_conf,
            min_hand_presence_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self._lm = HandLandmarker.create_from_options(opts)
        self._result = None
        self._t0 = time.monotonic()
        # Buffer reutilizável para evitar malloc a cada frame
        self._small = np.empty((PROC_H, PROC_W, 3), dtype=np.uint8)
        self._ov = None  # overlay pré-alocado

    def process(self, frame):
        cv2.resize(frame, (PROC_W, PROC_H), dst=self._small)
        rgb = cv2.cvtColor(self._small, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.monotonic() - self._t0) * 1000)
        self._result = self._lm.detect_for_video(mp_img, ts_ms)

    def get_hands(self, frame):
        h, w = frame.shape[:2]
        hands = []
        if not self._result or not self._result.hand_landmarks:
            return hands
        for i, lm_list in enumerate(self._result.hand_landmarks):
            label = self._result.handedness[i][0].category_name
            # Landmarks normalizados → coordenadas do frame COMPLETO
            landmarks = {
                idx: (int(lm.x * w), int(lm.y * h))
                for idx, lm in enumerate(lm_list)
            }
            hands.append({"label": label, "landmarks": landmarks})
        return hands

    def draw_landmarks(self, frame, hands):
        if not hands:
            return frame

        # Todos os glows num único overlay → 1 copy, 1 addWeighted
        if self._ov is None or self._ov.shape != frame.shape:
            self._ov = np.empty_like(frame)
        np.copyto(self._ov, frame)

        for hand in hands:
            lm = hand["landmarks"]
            for tip in (self.THUMB_TIP, self.INDEX_TIP):
                cv2.circle(self._ov, lm[tip], 16, (0, 220, 255), -1)

        cv2.addWeighted(self._ov, 0.25, frame, 0.75, 0, frame)

        # Elementos nítidos direto no frame (sem copy)
        for hand in hands:
            lm = hand["landmarks"]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, lm[a], lm[b], (0, 230, 180), 1, cv2.LINE_AA)
            for cx, cy in lm.values():
                cv2.circle(frame, (cx, cy), 3, (0, 200, 150), -1)
            for tip in (self.THUMB_TIP, self.INDEX_TIP):
                cx, cy = lm[tip]
                cv2.circle(frame, (cx, cy), 8, (0, 255, 230), -1)
                cv2.circle(frame, (cx, cy), 10, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def release(self):
        self._lm.close()
