"""
Efeitos extras do HoloFrame.

Cada efeito tem um atributo `enabled` (bool) que o painel liga/desliga.
"""

import collections
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
#  3. SCREENSHOT — captura automática depois de 2 s de moldura estável
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
