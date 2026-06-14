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
#  HELPERS COMPARTILHADOS
# ══════════════════════════════════════════════════════════════════════════════

def detect_pinch(hand, ratio: float = 0.55):
    """Retorna (is_pinch, ponto_medio_float32) de uma mão (polegar + indicador)."""
    lm    = hand["landmarks"]
    wrist = np.array(lm[0], dtype=np.float32)
    scale = np.linalg.norm(np.array(lm[9], dtype=np.float32) - wrist) + 1e-3
    tip_t = np.array(lm[4], dtype=np.float32)   # polegar
    tip_i = np.array(lm[8], dtype=np.float32)   # indicador
    mid   = (tip_t + tip_i) / 2.0
    return float(np.linalg.norm(tip_t - tip_i)) < scale * ratio, mid


def select_screen_region():
    """Seletor interativo de região da tela. Retorna (x, y, w, h) real ou None."""
    try:
        import mss as _mss
    except ImportError:
        print("AVISO: mss não instalado — execute: uv add mss")
        return None

    import ctypes
    hwnd = ctypes.windll.user32.FindWindowW(None, "HoloFrame")
    if hwnd:
        ctypes.windll.user32.ShowWindow(hwnd, 6)   # SW_MINIMIZE
        time.sleep(0.35)

    sct    = _mss.mss()
    mon0   = sct.monitors[0]
    shot   = cv2.cvtColor(np.array(sct.grab(mon0)), cv2.COLOR_BGRA2BGR)
    sh, sw = shot.shape[:2]
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

    WIN = "Selecione a tela  |  ENTER confirma  |  ESC cancela"
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
            dim = (canvas * 0.35).astype(np.uint8)
            dim[y0:y1, x0:x1] = canvas[y0:y1, x0:x1]
            canvas = dim
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 255), 2, cv2.LINE_AA)
            rw = max(1, int((x1 - x0) / scale))
            rh = max(1, int((y1 - y0) / scale))
            cv2.putText(canvas, f"{rw} x {rh} px", (x0 + 6, y0 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas,
                    "Clique e arraste  |  ENTER confirmar  |  ESC cancelar",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.imshow(WIN, canvas)

        key = cv2.waitKey(30) & 0xFF
        if key == 13 and p0 and p1 and p0 != p1:    # ENTER
            confirmed = True
            break
        elif key == 27:                              # ESC
            break

    cv2.destroyWindow(WIN)
    if hwnd:
        ctypes.windll.user32.ShowWindow(hwnd, 9)     # SW_RESTORE

    if confirmed and state["p0"] and state["p1"]:
        p0, p1 = state["p0"], state["p1"]
        x0, x1 = sorted([p0[0], p1[0]])
        y0, y1 = sorted([p0[1], p1[1]])
        rx = int(x0 / scale) + mon0["left"]
        ry = int(y0 / scale) + mon0["top"]
        rw = max(1, int((x1 - x0) / scale))
        rh = max(1, int((y1 - y0) / scale))
        return (rx, ry, rw, rh)
    return None


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
#  3. ROSTO ELÁSTICO — agarra o rosto com a pinça e estica como borracha
# ══════════════════════════════════════════════════════════════════════════════

class FaceWarp:
    """Agarra o rosto com a mão (pinça) e estica como borracha — Gomu Gomu.

    Fluxo:
      • O FaceTracker mapeia o rosto (478 pontos) → mostra a malha holográfica.
      • Ao fazer pinça (polegar + indicador) sobre o rosto, o ponto agarrado
        passa a seguir os dedos e os pixels ao redor são esticados via remap
        localizado, com perfil de queda suave (efeito borracha).
      • Soltar a pinça = o rosto volta ao normal.
    """

    PINCH_RATIO    = 0.55   # |polegar-indicador| / tamanho da mão p/ contar pinça
    SIGMA_RATIO    = 0.45   # σ do campo de deslocamento ∝ tamanho do rosto
    ROI_FACTOR     = 2.6    # raio da região processada = ROI_FACTOR * σ
    MAX_PULL       = 0.95   # deslocamento máx ∝ σ (abaixo do limite de dobra ≈1.16σ)
    PERP_SCALE     = 0.78   # achatamento perpendicular → estica em formato de faixa
    RELEASE_FRAMES = 3      # frames sem pinça antes de soltar (anti-flicker)

    def __init__(self):
        self.enabled = False
        self._lms    = None     # dict idx->(x,y) do rosto no frame atual
        self._box    = None     # (cx, cy, w, h)
        self._grab   = None     # {anchor, handle, sigma} enquanto agarrado
        self._miss   = 0        # frames consecutivos sem pinça
        try:
            import mediapipe as mp
            self._contours = mp.solutions.face_mesh.FACEMESH_CONTOURS
        except Exception:
            self._contours = None

    @property
    def name(self) -> str:
        if not self.enabled:
            return ""
        if self._grab is not None:
            return "PUXANDO"
        return "MAPEADO" if self._lms is not None else "..."

    # ------------------------------------------------------------------ entrada

    def set_face(self, landmarks):
        """Recebe landmarks do rosto (dict idx->(x,y)) ou None do FaceTracker."""
        self._lms = landmarks
        if landmarks:
            xs = [p[0] for p in landmarks.values()]
            ys = [p[1] for p in landmarks.values()]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            self._box = ((x0 + x1) // 2, (y0 + y1) // 2, x1 - x0, y1 - y0)
        else:
            self._box = None

    @staticmethod
    def _pinch(hand):
        return detect_pinch(hand, FaceWarp.PINCH_RATIO)

    def update(self, hands):
        if not self.enabled:
            self._grab = None
            return

        handle = None
        for hand in hands:
            pinching, mid = self._pinch(hand)
            if pinching:
                handle = mid
                break

        if handle is None:          # sem pinça neste frame
            if self._grab is not None:
                self._miss += 1
                if self._miss < self.RELEASE_FRAMES:
                    return          # segura o grab por alguns frames (anti-flicker)
            self._grab = None
            return
        self._miss = 0

        if self._grab is None:
            # precisa do rosto mapeado para COMEÇAR a agarrar (e sobre o rosto)
            if self._box is None:
                return
            cx, cy, fw, fh = self._box
            if abs(handle[0] - cx) > fw * 0.62 or abs(handle[1] - cy) > fh * 0.62:
                return
            sigma = max(fw, fh) * self.SIGMA_RATIO
            self._grab = {"anchor": handle.copy(), "sigma": sigma}

        # uma vez agarrado, segue os dedos mesmo se a detecção do rosto piscar
        self._grab["handle"] = handle

    # ------------------------------------------------------------------ warp

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.enabled or self._grab is None or "handle" not in self._grab:
            return frame

        anchor = self._grab["anchor"]
        handle = self._grab["handle"].copy()
        sigma  = self._grab["sigma"]

        d    = handle - anchor
        dist = float(np.linalg.norm(d))
        maxd = sigma * self.MAX_PULL
        if dist > maxd and dist > 1e-3:
            handle = anchor + d / dist * maxd

        return self._remap_pull(frame, anchor, handle, sigma)

    @classmethod
    def _remap_pull(cls, frame, anchor, handle, sigma):
        """Estica a pele do anchor até o handle com campo gaussiano anisotrópico.

        Gaussiano (suave) + deslocamento limitado a <1.16σ ⇒ sem dobra/duplicata.
        A queda mais estreita na perpendicular dá o formato de faixa (taffy).
        """
        h, w = frame.shape[:2]
        R = sigma * cls.ROI_FACTOR
        x0 = max(0, int(handle[0] - R)); x1 = min(w, int(handle[0] + R) + 1)
        y0 = max(0, int(handle[1] - R)); y1 = min(h, int(handle[1] + R) + 1)
        if x1 <= x0 or y1 <= y0:
            return frame

        xs = np.arange(x0, x1, dtype=np.float32)
        ys = np.arange(y0, y1, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        rx = gx - handle[0]
        ry = gy - handle[1]

        d = handle - anchor
        L = float(np.linalg.norm(d))
        if L > 1e-3:
            ux, uy = d[0] / L, d[1] / L              # direção do puxão
            along  = rx * ux + ry * uy               # ao longo do puxão
            perp   = -rx * uy + ry * ux              # perpendicular
            sp     = sigma * cls.PERP_SCALE
            wgt    = np.exp(-(along * along) / (sigma * sigma)
                            - (perp * perp) / (sp * sp))
        else:
            wgt = np.exp(-(rx * rx + ry * ry) / (sigma * sigma))

        map_x = gx - d[0] * wgt          # no dedo (wgt≈1): amostra do anchor
        map_y = gy - d[1] * wgt
        np.clip(map_x, 0, w - 1, out=map_x)
        np.clip(map_y, 0, h - 1, out=map_y)

        roi = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
        frame[y0:y1, x0:x1] = roi
        return frame

    # ------------------------------------------------------------------ overlay

    def draw(self, frame: np.ndarray, show_ui: bool = True):
        if not self.enabled or not show_ui:
            return        # modo limpo: o warp continua, sem malha/indicadores
        if self._grab is not None and "handle" in self._grab:
            self._draw_pull_fx(frame)
        elif self._lms is not None:
            self._draw_mesh(frame)
        self._draw_status(frame)

    def _draw_mesh(self, frame: np.ndarray):
        """Malha holográfica sobre o rosto — pontos (vetorizado) + contornos."""
        lm   = self._lms
        h, w = frame.shape[:2]
        pts  = np.array(list(lm.values()), dtype=np.int32)
        xs   = np.clip(pts[:, 0], 0, w - 1)
        ys   = np.clip(pts[:, 1], 0, h - 1)
        frame[ys, xs] = (0, 200, 150)          # 478 pontos em 1 op (sem 478 cv2.circle)
        if self._contours:
            for a, b in self._contours:
                if a in lm and b in lm:
                    cv2.line(frame, lm[a], lm[b], (0, 210, 160), 1, cv2.LINE_AA)

    def _draw_pull_fx(self, frame: np.ndarray):
        """Apenas o ponto de agarre sob os dedos — sem leque (estilo limpo)."""
        handle = self._grab["handle"]
        ci = (int(handle[0]), int(handle[1]))
        cv2.circle(frame, ci, 6, (40, 40, 255), -1)
        cv2.circle(frame, ci, 8, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_status(self, frame: np.ndarray):
        h = frame.shape[0]
        if self._grab is not None:
            col, label = (0, 255, 255), "PUXANDO"
        elif self._lms is not None:
            col, label = (0, 220, 170), "rosto mapeado - faca pinca para puxar"
        else:
            col, label = (0, 90, 160), "procurando rosto..."
        cv2.circle(frame, (18, h - 44), 5, col, -1)
        cv2.putText(frame, label, (30, h - 39),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)


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
        """Seleciona uma região da tela para exibir na moldura. True se confirmado."""
        region = select_screen_region()
        if region is None:
            return False
        if self._sct is None:
            import mss as _mss
            self._sct = _mss.mss()
        self.region      = region
        self.screen_mode = True
        self.enabled     = True
        print(f"Região de tela: {region[2]}×{region[3]}px")
        return True

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
#  5. TELA VIRTUAL — janelas do Windows ao vivo flutuando no espaço (estilo VR)
# ══════════════════════════════════════════════════════════════════════════════

class VirtualScreens:
    """Telas virtuais flutuantes com sensação de profundidade.

    Captura uma região/janela do Windows ao vivo (mss) e a renderiza como um
    painel flutuante com perspectiva + sombra + brilho. Cada painel tem alças
    dedicadas para manuseio com a pinça da mão — estilo headset VR.

      • N             captura uma nova tela (vira um painel ao vivo)
      • bola embaixo  pinça e arrasta → move o painel pelo espaço
      • grip no topo  pinça e arrasta → estica a borda de cima
      • grip na base  pinça e arrasta → estica a borda de baixo
      • X             remove o último painel
    """

    BASE_H         = 230   # altura base (px) do painel ao criar
    MIN_HH         = 55    # meia-altura mínima (px)
    MAX_HH         = 380   # meia-altura máxima (px)
    BALL_GAP       = 38    # distância da bola de mover abaixo da borda inferior
    GRAB_R           = 44     # raio de captura das alças (px)
    RELEASE_FRAMES   = 3      # frames sem pinça antes de soltar (anti-flicker)
    CAPTURE_INTERVAL = 0.045  # s entre capturas da tela (~22 Hz) → segura o FPS

    def __init__(self):
        self.enabled  = False
        self.panels   = []     # [{region, cx, cy, hh, aspect}]
        self._sct     = None
        self._grab    = None   # {idx, mode, ...}
        self._release = 0      # frames consecutivos sem pinça
        self._fw, self._fh = 1280, 720   # tamanho do frame (atualizado em update)

    @property
    def name(self) -> str:
        if not self.enabled:
            return ""
        n = len(self.panels)
        return f"{n} TELA{'S' if n != 1 else ''}" if n else "VAZIO"

    # ------------------------------------------------------------------ captura

    def spawn(self):
        """Abre o seletor de tela e cria um painel ao vivo com a região escolhida."""
        region = select_screen_region()
        if region is None:
            return
        if self._sct is None:
            try:
                import mss as _mss
                self._sct = _mss.mss()
            except ImportError:
                print("AVISO: mss não instalado — execute: uv add mss")
                return
        _, _, w, h = region
        self.panels.append({"region": region, "cx": None, "cy": None,
                            "hh": self.BASE_H / 2.0, "aspect": w / max(h, 1)})
        print(f"Tela virtual criada — {len(self.panels)} no total")

    def remove_last(self):
        if self.panels:
            self.panels.pop()
            self._grab = None

    # ------------------------------------------------------------------ geometria

    def _half_size(self, p):
        hh = p["hh"]
        return hh * p["aspect"], hh      # largura segue a altura → mantém proporção

    def _clamp_hh(self, hh, aspect):
        """Limita a meia-altura (e, via aspecto, a largura) a ~46% do frame."""
        hi = min(self.MAX_HH, self._fh * 0.46, self._fw * 0.46 / max(aspect, 1e-3))
        return float(np.clip(hh, self.MIN_HH, max(hi, self.MIN_HH)))

    def _hit(self, pt, c):
        return (pt[0] - c[0]) ** 2 + (pt[1] - c[1]) ** 2 <= self.GRAB_R ** 2

    def _handles(self, p, w_fr, h_fr):
        """Pontos (esquerda, direita, base, bola) das alças no espaço da tela."""
        quad  = self._quad(p, w_fr)
        left  = (quad[0] + quad[3]) / 2.0        # TL, BL
        right = (quad[1] + quad[2]) / 2.0        # TR, BR
        bot   = (quad[2] + quad[3]) / 2.0        # BR, BL
        ball  = bot + np.array([0.0, self.BALL_GAP], dtype=np.float32)
        ball[1] = min(ball[1], h_fr - 16)        # mantém a bola dentro da tela
        return left, right, bot, ball

    def _bring_front(self, idx):
        self.panels.append(self.panels.pop(idx))
        return len(self.panels) - 1

    # ------------------------------------------------------------------ interação

    def update(self, hands, frame_shape):
        if not self.enabled or not self.panels:
            self._grab = None
            return

        h_fr, w_fr = frame_shape[:2]
        self._fw, self._fh = w_fr, h_fr
        for i, p in enumerate(self.panels):
            if p["cx"] is None:
                p["cx"] = w_fr // 2 + (i - len(self.panels) // 2) * 80
                p["cy"] = h_fr // 2

        pts = [mid for ok, mid in (detect_pinch(hd) for hd in hands) if ok]
        if not pts:
            if self._grab is not None:
                self._release += 1
                if self._release < self.RELEASE_FRAMES:
                    return          # segura o grab por alguns frames (anti-flicker)
            self._grab = None
            return
        self._release = 0

        if self._grab is not None:
            self._apply_grab(pts[0])
        else:
            self._try_grab(pts[0], w_fr, h_fr)

    def _try_grab(self, pt, w_fr, h_fr):
        """Procura, do painel do topo p/ baixo, uma alça sob a pinça."""
        for i in range(len(self.panels) - 1, -1, -1):
            p = self.panels[i]
            if p["cx"] is None:
                continue
            left, right, _bot, ball = self._handles(p, w_fr, h_fr)
            if self._hit(pt, ball):
                p = self.panels[self._bring_front(i)]
                self._grab = {"idx": len(self.panels) - 1, "mode": "move",
                              "off": (p["cx"] - pt[0], p["cy"] - pt[1])}
                return
            if self._hit(pt, right):
                p  = self.panels[self._bring_front(i)]
                hw = p["hh"] * p["aspect"]
                self._grab = {"idx": len(self.panels) - 1, "mode": "right",
                              "fixed": p["cx"] - hw}          # borda esquerda fixa
                return
            if self._hit(pt, left):
                p  = self.panels[self._bring_front(i)]
                hw = p["hh"] * p["aspect"]
                self._grab = {"idx": len(self.panels) - 1, "mode": "left",
                              "fixed": p["cx"] + hw}          # borda direita fixa
                return
        self._grab = None

    def _apply_grab(self, pt):
        g = self._grab
        if g["idx"] >= len(self.panels):
            self._grab = None
            return
        p    = self.panels[g["idx"]]
        mode = g["mode"]
        if mode == "move":
            ox, oy = g["off"]
            p["cx"] = int(pt[0] + ox)
            p["cy"] = int(pt[1] + oy)
        elif mode == "right":
            lx = g["fixed"]                              # borda esquerda fixa
            hw = max((pt[0] - lx) / 2.0, 1.0)
            hh = self._clamp_hh(hw / p["aspect"], p["aspect"])
            p["hh"], p["cx"] = hh, int(lx + hh * p["aspect"])
        elif mode == "left":
            rx = g["fixed"]                              # borda direita fixa
            hw = max((rx - pt[0]) / 2.0, 1.0)
            hh = self._clamp_hh(hw / p["aspect"], p["aspect"])
            p["hh"], p["cx"] = hh, int(rx - hh * p["aspect"])

    # ------------------------------------------------------------------ render

    def draw(self, frame, show_ui: bool = True):
        if not self.enabled:
            return
        h_fr, w_fr = frame.shape[:2]
        for i, p in enumerate(self.panels):
            if p["cx"] is None:
                continue
            img = self._capture(p)               # captura da tela (com throttle)
            if img is not None:
                self._render_panel(frame, p, img)
            if show_ui:
                active = (self._grab["mode"]
                          if self._grab and self._grab["idx"] == i else None)
                self._draw_handles(frame, p, w_fr, h_fr, active)
        if show_ui:
            self._draw_status(frame)

    def _capture(self, p):
        """Captura a região da tela com throttle — reusa o último frame entre
        capturas para segurar o FPS (a tela não precisa atualizar a 30/60 Hz)."""
        now = time.time()
        if p.get("_img") is not None and now - p.get("_t", 0.0) < self.CAPTURE_INTERVAL:
            return p["_img"]
        img = self._grab_region(p["region"])
        if img is not None:
            p["_img"], p["_t"] = img, now
        return p.get("_img")

    def _grab_region(self, region):
        if self._sct is None:
            return None
        try:
            x, y, w, h = region
            shot = self._sct.grab({"left": x, "top": y, "width": w, "height": h})
            return cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)
        except Exception:
            return None

    def _quad(self, p, w_fr=None):
        """Retângulo reto do painel (sem perspectiva) — a profundidade vem do
        tamanho + sombra + escurecimento, mantendo a tela sempre alinhada."""
        cx, cy = p["cx"], p["cy"]
        hw, hh = self._half_size(p)
        return np.array([[cx - hw, cy - hh], [cx + hw, cy - hh],
                         [cx + hw, cy + hh], [cx - hw, cy + hh]], dtype=np.float32)

    def _render_panel(self, frame, p, img):
        h_fr, w_fr = frame.shape[:2]
        quad = self._quad(p, w_fr)

        bx0 = max(0,    int(np.floor(quad[:, 0].min())) - 16)
        by0 = max(0,    int(np.floor(quad[:, 1].min())) - 18)
        bx1 = min(w_fr, int(np.ceil(quad[:, 0].max()))  + 16)
        by1 = min(h_fr, int(np.ceil(quad[:, 1].max()))  + 20)
        if bx1 - bx0 < 4 or by1 - by0 < 4:
            return

        sub     = frame[by0:by1, bx0:bx1]
        sh, sw  = sub.shape[:2]
        local   = quad - np.array([bx0, by0], dtype=np.float32)

        # sombra deslocada → sensação de flutuar
        shadow = (local + np.array([9, 13], dtype=np.float32)).astype(np.int32)
        ov = sub.copy()
        cv2.fillConvexPoly(ov, shadow, (0, 0, 0), cv2.LINE_AA)
        cv2.addWeighted(ov, 0.40, sub, 0.60, 0, sub)

        # warp da janela capturada para o quad
        ih, iw = img.shape[:2]
        src = np.array([[0, 0], [iw, 0], [iw, ih], [0, ih]], dtype=np.float32)
        M       = cv2.getPerspectiveTransform(src, local)
        warped  = cv2.warpPerspective(img, M, (sw, sh), flags=cv2.INTER_LINEAR)
        mask    = np.zeros((sh, sw), np.uint8)          # mais barato que 2º warp
        cv2.fillConvexPoly(mask, local.astype(np.int32), 255)

        # profundidade: painel menor (mais longe) = mais escuro
        ratio = (p["hh"] - self.MIN_HH) / max(self.MAX_HH - self.MIN_HH, 1)
        dim   = float(np.clip(0.55 + 0.45 * ratio, 0.45, 1.0))
        if dim < 0.99:
            warped = (warped.astype(np.float32) * dim).astype(np.uint8)

        m = mask > 0
        sub[m] = warped[m]

        # borda holográfica
        cv2.polylines(sub, [local.astype(np.int32)], True, (0, 230, 255), 2, cv2.LINE_AA)

    # ------------------------------------------------------------------ alças

    def _draw_handles(self, frame, p, w_fr, h_fr, active):
        left, right, bot, ball = self._handles(p, w_fr, h_fr)
        self._grip(frame, left,  "left",  active=(active == "left"))
        self._grip(frame, right, "right", active=(active == "right"))

        bx, by = int(ball[0]), int(ball[1])
        # haste ligando o painel à bola de mover
        cv2.line(frame, (int(bot[0]), int(bot[1])), (bx, by),
                 (0, 170, 210), 2, cv2.LINE_AA)
        on  = active == "move"
        r   = 15 if on else 12
        col = (0, 255, 255) if on else (0, 210, 240)
        cv2.circle(frame, (bx, by), r + 2, (12, 26, 36), -1)
        cv2.circle(frame, (bx, by), r, col, -1)
        cv2.circle(frame, (bx, by), r, (255, 255, 255), 1, cv2.LINE_AA)
        a = r - 5      # ícone de mover (cruz)
        cv2.line(frame, (bx - a, by), (bx + a, by), (15, 28, 38), 2, cv2.LINE_AA)
        cv2.line(frame, (bx, by - a), (bx, by + a), (15, 28, 38), 2, cv2.LINE_AA)

    def _grip(self, frame, c, side, active):
        cx, cy = int(c[0]), int(c[1])
        w2, h2 = 7, 30          # barra vertical na lateral
        col    = (0, 255, 255) if active else (0, 205, 235)
        cv2.rectangle(frame, (cx - w2, cy - h2), (cx + w2, cy + h2), (12, 26, 36), -1)
        cv2.rectangle(frame, (cx - w2, cy - h2), (cx + w2, cy + h2), col, 1, cv2.LINE_AA)
        # chevron apontando para fora (direção de aumentar)
        out    = -1 if side == "left" else 1
        tip    = (cx + out * (w2 + 7), cy)
        base_x = cx + out * w2
        cv2.line(frame, (base_x, cy - 8), tip, col, 2, cv2.LINE_AA)
        cv2.line(frame, (base_x, cy + 8), tip, col, 2, cv2.LINE_AA)

    def _draw_status(self, frame):
        h = frame.shape[0]
        if not self.panels:
            col, label = (0, 150, 210), "N: capturar tela virtual"
        else:
            col, label = (0, 230, 255), "bola: mover  |  laterais: tamanho  |  X: remover"
        cv2.circle(frame, (18, h - 64), 5, col, -1)
        cv2.putText(frame, label, (30, h - 59),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)


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
