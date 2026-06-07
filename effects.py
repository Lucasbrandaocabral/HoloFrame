"""
Efeitos extras do HoloFrame.

Cada efeito tem um atributo `enabled` (bool) que o painel liga/desliga.
"""

import collections
import math
import os
import time

import cv2
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  1. TRILHA DE LUZ — rastro neon nas pontas dos indicadores
# ══════════════════════════════════════════════════════════════════════════════

class LightTrail:
    MAX = 40

    def __init__(self):
        self.enabled = True
        self._trails: dict[int, collections.deque] = {}

    def update(self, hands):
        seen = set()
        for i, hand in enumerate(hands):
            tip = hand["landmarks"][8]   # index tip
            if i not in self._trails:
                self._trails[i] = collections.deque(maxlen=self.MAX)
            self._trails[i].append(tip)
            seen.add(i)
        for k in list(self._trails):
            if k not in seen:
                del self._trails[k]

    def draw(self, frame):
        if not self.enabled:
            return
        palettes = [
            lambda a: (int(a * 80), int(a * 200), 255),   # hand 0 → cyan
            lambda a: (int(a * 200), int(a * 60), 255),   # hand 1 → magenta
        ]
        for i, deque in self._trails.items():
            pts = list(deque)
            if len(pts) < 2:
                continue
            n = len(pts)
            color_fn = palettes[i % len(palettes)]
            for j in range(1, n):
                a = j / n
                thick = max(1, int(a * 6))
                cv2.line(frame, pts[j - 1], pts[j], color_fn(a), thick, cv2.LINE_AA)
            # bright dot at tip
            cv2.circle(frame, pts[-1], 5, (255, 255, 255), -1)


# ══════════════════════════════════════════════════════════════════════════════
#  2. GLITCH — distorção quando a moldura se move rápido
# ══════════════════════════════════════════════════════════════════════════════

class GlitchEffect:
    def __init__(self):
        self.enabled = True
        self._prev = None
        self._frames_left = 0

    def update(self, corners):
        if corners is None:
            self._prev = None
            self._frames_left = max(0, self._frames_left - 1)
            return
        if self._prev is not None and np.linalg.norm(corners - self._prev) > 11:
            self._frames_left = 10
        self._prev = corners.copy()
        self._frames_left = max(0, self._frames_left - 1)

    @property
    def is_active(self):
        return self.enabled and self._frames_left > 0


def apply_glitch(img: np.ndarray) -> np.ndarray:
    """Aplica channel-shift + faixas de ruído horizontal à imagem."""
    sx = np.random.randint(-12, 12)
    out = img.copy()
    out[:, :, 2] = np.roll(img[:, :, 2],  sx, axis=1)   # R shift →
    out[:, :, 0] = np.roll(img[:, :, 0], -sx, axis=1)   # B shift ←
    h = out.shape[0]
    for _ in range(np.random.randint(3, 8)):
        y  = np.random.randint(0, h)
        bh = np.random.randint(2, 12)
        out[y:y + bh] = np.roll(out[y:y + bh], np.random.randint(-35, 35), axis=1)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  3. CAPTURA DE ROSTO — detecta e enquadra o rosto na moldura
# ══════════════════════════════════════════════════════════════════════════════

class FaceCapture:
    PAD     = 0.45
    EVERY_N = 2   # detecta a cada 2 frames

    def __init__(self):
        self.enabled   = False
        self._face_img = None
        self._detector = None
        self._counter  = 0
        self._last_box = None
        self._init_detector()

    def _init_detector(self):
        xml = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        c = cv2.CascadeClassifier(xml)
        if not c.empty():
            self._detector = c
            print("FaceCapture: cascade carregado →", xml)
        else:
            print("AVISO: cascade não encontrado →", xml)

    @property
    def name(self) -> str:
        return "LUFFY" if self._face_img is not None else ""

    def update(self, frame: np.ndarray):
        if not self.enabled or self._detector is None:
            return
        self._counter += 1
        if self._counter % self.EVERY_N != 0:
            if self._last_box is not None:
                self._crop(frame, *self._last_box)
            return

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.equalizeHist(gray, gray)   # melhora contraste para detectar melhor
        faces = self._detector.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40))

        if len(faces) == 0:
            return

        x, y, bw, bh = max(faces, key=lambda f: f[2] * f[3])
        self._last_box = (x, y, bw, bh)
        self._crop(frame, x, y, bw, bh)

    def _crop(self, frame: np.ndarray, x: int, y: int, w: int, h: int):
        fh, fw = frame.shape[:2]
        pad = int(max(w, h) * self.PAD)
        x1 = max(0, x - pad)
        y1 = max(0, y - int(pad * 1.4))
        x2 = min(fw, x + w + pad)
        y2 = min(fh, y + h + pad)
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            self._face_img = crop.copy()

    def get_frame(self) -> "np.ndarray | None":
        if not self.enabled or self._face_img is None:
            return None
        return self._luffy_stretch(self._face_img)

    def _luffy_stretch(self, img: np.ndarray) -> np.ndarray:
        h, w  = img.shape[:2]
        phase = math.sin(time.time() * 2.8)

        sx = 1.0 + phase * 0.55   # stretch horizontal (0.45→1.55)
        sy = 1.0 - phase * 0.30   # squish vertical inverso

        cx, cy = w / 2.0, h / 2.0
        dx = np.tile(np.arange(w, dtype=np.float32), (h, 1))
        dy = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w))

        map_x = np.clip(cx + (dx - cx) / max(sx, 0.05), 0, w - 1).astype(np.float32)
        map_y = np.clip(cy + (dy - cy) / max(sy, 0.05), 0, h - 1).astype(np.float32)

        return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

    def draw(self, frame: np.ndarray):
        if not self.enabled:
            return
        h_fr = frame.shape[0]
        col   = (0, 255, 200) if self._face_img is not None else (0, 80, 160)
        label = "LUFFY ativo" if self._face_img is not None else "procurando rosto..."
        cv2.circle(frame, (18, h_fr - 44), 5, col, -1)
        cv2.putText(frame, label, (30, h_fr - 39),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

        # debug: retângulo do rosto detectado
        if self._last_box is not None:
            x, y, w, h = self._last_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 180), 2, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  3b. SCREENSHOT — captura automática depois de 2 s de moldura estável
# ══════════════════════════════════════════════════════════════════════════════

class ScreenshotCapture:
    HOLD     = 2.0   # segundos parado para disparar
    COOLDOWN = 3.0   # intervalo mínimo entre capturas

    def __init__(self):
        self.enabled   = True
        self._hold_start = None
        self._prev       = None
        self._last_save  = 0.0
        self._flash      = 0
        self._msg        = ""
        self._msg_frames = 0
        os.makedirs("screenshots", exist_ok=True)

    def update(self, frame: np.ndarray, corners):
        if not self.enabled or corners is None:
            self._hold_start = None
            self._prev = corners
            return

        moved = (self._prev is not None and
                 np.linalg.norm(corners - self._prev) > 9)
        self._prev = corners.copy()

        if moved:
            self._hold_start = None
            return

        if self._hold_start is None:
            self._hold_start = time.time()

        elapsed = time.time() - self._hold_start
        if elapsed >= self.HOLD and time.time() - self._last_save > self.COOLDOWN:
            fname = f"screenshots/holoframe_{int(time.time())}.png"
            cv2.imwrite(fname, frame)
            self._last_save  = time.time()
            self._hold_start = time.time() + 99   # evita re-trigger imediato
            self._flash      = 12
            self._msg        = f"SAVED  {os.path.basename(fname)}"
            self._msg_frames = 90

    def progress(self):
        """Retorna 0-1 do progresso até captura, ou None se inativo."""
        if not self.enabled or self._hold_start is None:
            return None
        return min(1.0, (time.time() - self._hold_start) / self.HOLD)

    def draw(self, frame: np.ndarray, corners):
        # Flash branco
        if self._flash > 0:
            a = self._flash / 12
            np.clip(frame.astype(np.int32) + int(a * 210), 0, 255,
                    out=frame.astype(np.int32))   # in-place add
            # simpler but correct:
            bright = np.clip(frame.astype(np.int32) + int(a * 210), 0, 255).astype(np.uint8)
            frame[:] = bright
            self._flash -= 1

        # Mensagem de confirmação
        if self._msg_frames > 0:
            h, w = frame.shape[:2]
            a = min(1.0, self._msg_frames / 25)
            col = (0, int(255 * a), int(140 * a))
            cv2.putText(frame, self._msg, (w // 2 - 200, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)
            self._msg_frames -= 1

        # Anel de progresso no canto TL da moldura
        if corners is not None:
            p = self.progress()
            if p is not None and 0 < p < 1.0:
                cx, cy = int(corners[0][0]), int(corners[0][1])
                cv2.ellipse(frame, (cx, cy), (24, 24), -90,
                            0, int(p * 360), (0, 255, 120), 3, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  4. ESPELHO / TELA — webcam ou região da tela dentro da moldura
# ══════════════════════════════════════════════════════════════════════════════

class MirrorMode:
    def __init__(self):
        self.enabled     = False
        self.screen_mode = False   # False = webcam, True = captura de tela
        self.region      = None    # (x, y, w, h) em coords reais de tela
        self._sct        = None    # instância mss

    @property
    def name(self) -> str:
        if not self.enabled:
            return ""
        return "TELA" if self.screen_mode else "WEBCAM"

    # ------------------------------------------------------------------ public

    def get_frame(self, webcam_prev: "np.ndarray | None") -> "np.ndarray | None":
        """Retorna o frame a usar na moldura holográfica."""
        if not self.enabled:
            return None
        if self.screen_mode:
            return self._capture_region()
        return webcam_prev   # comportamento antigo (webcam)

    def select_region(self) -> bool:
        """Abre seletor interativo de região da tela. Retorna True se confirmado."""
        try:
            import mss as _mss
        except ImportError:
            print("AVISO: mss não instalado — execute: uv add mss")
            return False

        # Minimiza HoloFrame para não aparecer no screenshot
        import ctypes
        hwnd = ctypes.windll.user32.FindWindowW(None, "HoloFrame")
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 6)   # SW_MINIMIZE
            time.sleep(0.35)

        if self._sct is None:
            self._sct = _mss.mss()

        # Captura todos os monitores combinados
        mon0  = self._sct.monitors[0]
        shot  = np.array(self._sct.grab(mon0))
        shot  = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
        sh, sw = shot.shape[:2]

        # Redimensiona para caber em janela de seleção (máx 1400×880)
        scale  = min(1400 / sw, 880 / sh, 1.0)
        dw, dh = int(sw * scale), int(sh * scale)
        base   = cv2.resize(shot, (dw, dh)) if scale < 1.0 else shot.copy()

        state = {"down": False, "p0": None, "p1": None}

        def on_mouse(evt, x, y, flags, _param):
            if evt == cv2.EVENT_LBUTTONDOWN:
                state.update({"down": True, "p0": (x, y), "p1": (x, y)})
            elif evt == cv2.EVENT_MOUSEMOVE and state["down"]:
                state["p1"] = (x, y)
            elif evt == cv2.EVENT_LBUTTONUP:
                state["down"] = False
                state["p1"]   = (x, y)

        WIN = "Selecione a regiao  |  ENTER confirma  |  ESC cancela"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, dw, dh)
        cv2.setMouseCallback(WIN, on_mouse)

        confirmed = False
        while True:
            canvas = base.copy()
            p0, p1 = state["p0"], state["p1"]
            if p0 and p1 and p0 != p1:
                x0, x1 = sorted([p0[0], p1[0]])
                y0, y1 = sorted([p0[1], p1[1]])
                # escurece fora da seleção
                dim = (canvas * 0.35).astype(np.uint8)
                dim[y0:y1, x0:x1] = canvas[y0:y1, x0:x1]
                canvas = dim
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 255), 2, cv2.LINE_AA)
                rw = max(1, int((x1 - x0) / scale))
                rh = max(1, int((y1 - y0) / scale))
                cv2.putText(canvas, f"{rw} x {rh} px", (x0 + 6, y0 + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1, cv2.LINE_AA)

            cv2.putText(canvas,
                        "Clique e arraste para selecionar  |  ENTER confirmar  |  ESC cancelar",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)
            cv2.imshow(WIN, canvas)

            key = cv2.waitKey(30) & 0xFF
            if key == 13 and p0 and p1 and p0 != p1:   # ENTER
                confirmed = True
                break
            elif key == 27:                              # ESC
                break

        cv2.destroyWindow(WIN)

        # Restaura HoloFrame
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 9)   # SW_RESTORE

        if confirmed and state["p0"] and state["p1"]:
            p0, p1 = state["p0"], state["p1"]
            x0, x1 = sorted([p0[0], p1[0]])
            y0, y1 = sorted([p0[1], p1[1]])
            rx = int(x0 / scale) + mon0["left"]
            ry = int(y0 / scale) + mon0["top"]
            rw = max(1, int((x1 - x0) / scale))
            rh = max(1, int((y1 - y0) / scale))
            self.region      = (rx, ry, rw, rh)
            self.screen_mode = True
            self.enabled     = True
            print(f"Região de tela: x={rx} y={ry} {rw}×{rh}px")
            return True

        return False

    # ------------------------------------------------------------------ private

    def _capture_region(self) -> "np.ndarray | None":
        if self._sct is None or self.region is None:
            return None
        try:
            x, y, w, h = self.region
            img = np.array(self._sct.grab({"left": x, "top": y, "width": w, "height": h}))
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        except Exception:
            return None


# ══════════════════════════════════════════════════════════════════════════════
#  5. FILTRO DE COR — cicla entre filtros visuais (tecla 5 avança)
# ══════════════════════════════════════════════════════════════════════════════

_SEPIA_KERNEL = np.array(
    [[0.131, 0.534, 0.272],
     [0.168, 0.686, 0.349],
     [0.189, 0.769, 0.393]],
    dtype=np.float32,
)

FILTER_NAMES = ["NORMAL", "NEON", "P&B", "SEPIA", "INFRAVERMELHO"]


class ColorFilter:
    def __init__(self):
        self.enabled = False
        self._idx    = 0   # 0 = NORMAL (sem efeito)

    @property
    def name(self) -> str:
        return FILTER_NAMES[self._idx]

    def cycle(self):
        self._idx = (self._idx + 1) % len(FILTER_NAMES)
        self.enabled = (self._idx != 0)

    def apply(self, img: np.ndarray) -> np.ndarray:
        if not self.enabled or img is None or self._idx == 0:
            return img
        name = FILTER_NAMES[self._idx]

        if name == "NEON":
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 2.4, 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)
            return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        if name == "P&B":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if name == "SEPIA":
            return np.clip(cv2.transform(img, _SEPIA_KERNEL), 0, 255).astype(np.uint8)

        if name == "INFRAVERMELHO":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)

        return img


# ══════════════════════════════════════════════════════════════════════════════
#  6. EXPLOSÃO — punho fechado dispara burst de partículas
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  8. DESENHO NO AR — indicador esticado desenha neon; paz ✌ troca cor; C limpa
# ══════════════════════════════════════════════════════════════════════════════

class AirDraw:
    COLORS = [
        (0,   255, 255),   # cyan
        (255,  80, 255),   # magenta
        (80,  255, 100),   # verde
        (255, 160,   0),   # laranja
        (130, 100, 255),   # azul
        (255, 255,  80),   # amarelo
    ]
    COLOR_NAMES = ["CYAN", "MAGENTA", "VERDE", "LARANJA", "AZUL", "AMARELO"]

    def __init__(self):
        self.enabled       = False
        self._canvas       = None   # camada de desenho persistente
        self._prev_pt      = None
        self._color_idx    = 0
        self._size         = 3
        self._peace_locked = False  # evita trocar cor em todo frame do gesto

    @property
    def color(self):
        return self.COLORS[self._color_idx]

    @property
    def name(self) -> str:
        return self.COLOR_NAMES[self._color_idx]

    def clear(self):
        if self._canvas is not None:
            self._canvas[:] = 0

    # ------------------------------------------------------------------ gestos

    @staticmethod
    def _gesture(hand) -> str:
        """'draw' | 'color' | 'none'"""
        lm       = hand["landmarks"]
        idx_up   = lm[8][1]  < lm[6][1]    # indicador esticado
        mid_up   = lm[12][1] < lm[10][1]   # médio esticado
        ring_dn  = lm[16][1] > lm[14][1]   # anelar dobrado
        pinky_dn = lm[20][1] > lm[18][1]   # mindinho dobrado
        if idx_up and not mid_up and ring_dn and pinky_dn:
            return "draw"
        if idx_up and mid_up and ring_dn and pinky_dn:
            return "color"
        return "none"

    # ------------------------------------------------------------------ update / draw

    def update(self, hands, frame_shape):
        if not self.enabled:
            self._prev_pt = None
            return

        h, w = frame_shape[:2]
        if self._canvas is None or self._canvas.shape[:2] != (h, w):
            self._canvas = np.zeros((h, w, 3), dtype=np.uint8)

        gesture = "none"
        tip_pt  = None
        for hand in hands:
            g = self._gesture(hand)
            if g != "none":
                gesture, tip_pt = g, hand["landmarks"][8]
                break

        # troca de cor — bloqueado até soltar o gesto
        if gesture == "color" and not self._peace_locked:
            self._color_idx    = (self._color_idx + 1) % len(self.COLORS)
            self._peace_locked = True
        elif gesture != "color":
            self._peace_locked = False

        # desenho
        if gesture == "draw" and tip_pt is not None:
            if self._prev_pt is not None:
                glow = tuple(c // 4 for c in self.color)
                cv2.line(self._canvas, self._prev_pt, tip_pt,
                         glow, self._size * 5, cv2.LINE_AA)
                cv2.line(self._canvas, self._prev_pt, tip_pt,
                         self.color, self._size, cv2.LINE_AA)
            cv2.circle(self._canvas, tip_pt, self._size + 2, self.color, -1)
            self._prev_pt = tip_pt
        else:
            self._prev_pt = None

    def draw(self, frame: np.ndarray):
        if not self.enabled:
            return
        if self._canvas is not None:
            cv2.add(frame, self._canvas, dst=frame)   # adição saturada = glow neon

        # indicador de cor no canto inferior esquerdo
        h = frame.shape[0]
        cv2.circle(frame, (18, h - 20), 7, self.color, -1)
        cv2.circle(frame, (18, h - 20), 9, self.color, 1, cv2.LINE_AA)
        cv2.putText(frame, "paz=cor   C=limpar",
                    (32, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.36, (70, 110, 130), 1, cv2.LINE_AA)

# ══════════════════════════════════════════════════════════════════════════════
#  7. CLONE FANTASMA — palma aberta por 1 s congela o frame; sai do lugar e
#     você vê dois de si mesmo na tela (dupla exposição)
# ══════════════════════════════════════════════════════════════════════════════

class GhostClone:
    HOLD = 0.9   # segundos de palma aberta para disparar

    def __init__(self):
        self.enabled  = False
        self.frozen   = None    # frame congelado
        self._palm_t  = None    # quando a palma foi detectada
        self._locked  = False   # evita re-trigger enquanto palma continua aberta

    @property
    def name(self) -> str:
        return "CONGELADO" if self.frozen is not None else ""

    # ------------------------------------------------------------------ detecção

    @staticmethod
    def _is_open_palm(hand) -> bool:
        lm    = hand["landmarks"]
        wrist = np.array(lm[0])
        scale = np.linalg.norm(np.array(lm[9]) - wrist)
        if scale < 1:
            return False
        thresh = scale * 1.45
        return all(
            np.linalg.norm(np.array(lm[t]) - wrist) > thresh
            for t in (4, 8, 12, 16, 20)
        )

    # ------------------------------------------------------------------ update / apply

    def update(self, hands, frame: np.ndarray):
        if not self.enabled:
            return
        is_palm = any(self._is_open_palm(h) for h in hands)

        if is_palm and not self._locked:
            if self._palm_t is None:
                self._palm_t = time.time()
            elif time.time() - self._palm_t >= self.HOLD:
                self.frozen  = None if self.frozen is not None else frame.copy()
                self._locked = True   # precisa tirar a mão para disparar de novo
        elif not is_palm:
            self._palm_t = None
            self._locked = False

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Dupla exposição: congelado 60% + ao vivo 60% → ambos visíveis."""
        if not self.enabled or self.frozen is None:
            return frame
        if self.frozen.shape != frame.shape:
            return frame
        return cv2.addWeighted(self.frozen, 0.60, frame, 0.60, 0)

    def draw_indicator(self, frame: np.ndarray, hands):
        """Anel de progresso sobre a palma enquanto aguarda o disparo."""
        if not self.enabled or self._palm_t is None or self._locked:
            return
        p = min(1.0, (time.time() - self._palm_t) / self.HOLD)
        for hand in hands:
            if self._is_open_palm(hand):
                cx, cy = hand["landmarks"][9]
                label  = "LIBERA" if self.frozen is not None else "CONGELA"
                cv2.ellipse(frame, (cx, cy), (32, 32), -90,
                            0, int(p * 360), (0, 255, 200), 3, cv2.LINE_AA)
                cv2.putText(frame, label, (cx - 32, cy - 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 200), 1, cv2.LINE_AA)

class ExplosionEffect:
    def __init__(self):
        self.enabled    = True
        self._particles = []
        self._was_fist  = False

    @staticmethod
    def _is_fist(hand) -> bool:
        lm   = hand["landmarks"]
        palm = np.array(lm[9])   # middle MCP
        return all(
            np.linalg.norm(np.array(lm[t]) - palm) < 68
            for t in (8, 12, 16, 20)
        )

    def update(self, hands, corners):
        if not self.enabled:
            return
        is_fist = any(self._is_fist(h) for h in hands)
        if is_fist and not self._was_fist and corners is not None:
            self._burst(corners)
        self._was_fist = is_fist

        alive = []
        for p in self._particles:
            p["x"] += p["vx"];  p["y"] += p["vy"]
            p["vx"] *= 0.93;    p["vy"] = p["vy"] * 0.93 + 0.28
            p["life"] -= p["decay"]
            if p["life"] > 0:
                alive.append(p)
        self._particles = alive

    def _burst(self, corners):
        cx, cy  = corners.mean(axis=0)
        palette = [
            (0, 255, 255), (0, 120, 255), (255, 80, 0),
            (0, 255, 100), (255, 0, 160), (200, 200, 255),
        ]
        for _ in range(160):
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(4, 22)
            self._particles.append({
                "x": cx, "y": cy,
                "vx": np.cos(angle) * speed,
                "vy": np.sin(angle) * speed,
                "life":  1.0,
                "decay": np.random.uniform(0.010, 0.032),
                "size":  np.random.randint(3, 11),
                "color": palette[np.random.randint(len(palette))],
            })

    def draw(self, frame: np.ndarray):
        if not self._particles:
            return
        h, w = frame.shape[:2]
        for p in self._particles:
            x, y = int(p["x"]), int(p["y"])
            if 0 <= x < w and 0 <= y < h:
                s = max(1, int(p["size"] * p["life"]))
                cv2.circle(frame, (x, y), s, p["color"], -1)
                if s > 3:
                    bright = tuple(min(255, c + 120) for c in p["color"])
                    cv2.circle(frame, (x, y), s + 2, bright, 1, cv2.LINE_AA)
