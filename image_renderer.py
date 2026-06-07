import math
import time

import cv2
import numpy as np


class ImageRenderer:
    ANIM_SPEED    = 0.10
    SCAN_SPEED    = 0.007
    MAX_PARTICLES = 70

    def __init__(self, image_path=None):
        self.image = None
        self.load_image(image_path)

        self._alpha  = 0.0
        self._target = 0.0
        self._last_corners = None

        self._scan       = 0.0
        self._t0         = time.time()
        self._particles  = []

        # Buffers pré-alocados
        self._ov      = None
        self._mask_u8 = None

        # ── Atributos que main.py define a cada frame ──────────────────────
        self.source_frame: np.ndarray | None = None  # mirror mode
        self.zoom_factor:  float             = 1.0
        self.glitch_active: bool             = False
        self.color_filter                    = None  # ref ao ColorFilter

    # ------------------------------------------------------------------ public

    def load_image(self, path=None):
        if path:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                if img.ndim == 3 and img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                self.image = img
                return
        self.image = self._make_default()

    def set_active(self, active: bool):
        self._target = 1.0 if active else 0.0

    def render(self, frame: np.ndarray, corners=None) -> np.ndarray:
        self._tick(corners)

        if self._alpha < 0.005:
            self._particles.clear()
            return frame
        if self._last_corners is None:
            return frame

        pts = self._last_corners

        frame = self._warp_image(frame, pts)
        self._draw_hud(frame, pts)
        self._draw_particles(frame)

        return frame

    # ------------------------------------------------------------------ tick

    def _tick(self, corners):
        self._alpha += (self._target - self._alpha) * self.ANIM_SPEED
        self._alpha  = float(np.clip(self._alpha, 0.0, 1.0))

        if corners is not None:
            self._last_corners = corners.astype(np.float32)

        self._scan = (self._scan + self.SCAN_SPEED) % 1.0

        self._particles.clear()

    # ------------------------------------------------------------------ warp

    def _warp_image(self, frame: np.ndarray, pts: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]

        # Escolhe a fonte: mirror (webcam) ou imagem estática
        src_img = self.source_frame if self.source_frame is not None else self.image

        # Aplica zoom (crop + resize)
        if self.zoom_factor > 1.02 or self.zoom_factor < 0.98:
            src_img = self._zoom(src_img, self.zoom_factor)

        # Aplica filtro de cor
        if self.color_filter is not None:
            src_img = self.color_filter.apply(src_img)

        ih, iw = src_img.shape[:2]
        src = np.array([[0, 0], [iw, 0], [iw, ih], [0, ih]], dtype=np.float32)

        try:
            M = cv2.getPerspectiveTransform(src, pts.astype(np.float32))
        except cv2.error:
            return frame

        warped = cv2.warpPerspective(src_img, M, (w, h))

        # Efeitos pós-warp
        if self.glitch_active:
            from effects import apply_glitch
            warped = apply_glitch(warped)

        # Máscara pré-alocada
        if self._mask_u8 is None or self._mask_u8.shape != (h, w):
            self._mask_u8 = np.zeros((h, w), dtype=np.uint8)
        else:
            self._mask_u8[:] = 0
        cv2.fillPoly(self._mask_u8, [pts.astype(np.int32)], 255)

        # Blend dentro do quad
        idx = self._mask_u8 > 0
        if self._alpha >= 0.995:
            frame[idx] = warped[idx]
        else:
            a = self._alpha
            frame[idx] = np.clip(
                warped[idx].astype(np.float32) * a +
                frame[idx].astype(np.float32) * (1.0 - a),
                0, 255,
            ).astype(np.uint8)

        return frame

    def _zoom(self, img: np.ndarray, factor: float) -> np.ndarray:
        factor = float(np.clip(factor, 0.25, 4.0))
        h, w   = img.shape[:2]
        nw, nh = max(10, int(w / factor)), max(10, int(h / factor))
        if factor >= 1.0:
            x1, y1 = (w - nw) // 2, (h - nh) // 2
            return cv2.resize(img[y1:y1 + nh, x1:x1 + nw], (w, h),
                              interpolation=cv2.INTER_LINEAR)
        else:
            small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            out   = np.zeros((h, w, 3), dtype=np.uint8)
            x1, y1 = (w - nw) // 2, (h - nh) // 2
            out[y1:y1 + nh, x1:x1 + nw] = small
            return out

    def _scanline(self, img: np.ndarray):
        h  = img.shape[0]
        sy = int(self._scan * h)
        for dy, s in ((-2, 14), (-1, 28), (0, 52), (1, 28), (2, 14)):
            y = sy + dy
            if 0 <= y < h:
                img[y] = np.clip(img[y].astype(np.int32) + s, 0, 255)

    # ------------------------------------------------------------------ HUD (1 overlay total)

    def _draw_hud(self, frame: np.ndarray, pts: np.ndarray):
        pass   # modo limpo: sem bordas, brilhos ou decorações

    def _draw_corners(self, frame, pts, t):
        pulse   = int(math.sin(t * 2.5) * 3)
        bracket = 30 + pulse
        center  = pts.mean(axis=0)
        colors  = [(0, 255, 255), (0, 180, 255), (0, 255, 200), (80, 150, 255)]
        for i, (cx, cy) in enumerate(pts):
            dx = 1 if cx >= center[0] else -1
            dy = 1 if cy >= center[1] else -1
            col = colors[i]
            cv2.line(frame, (cx, cy), (cx + dx * bracket, cy), col, 3, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + dy * bracket), col, 3, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
            cv2.circle(frame, (cx, cy), 9, col, 1, cv2.LINE_AA)

    def _draw_grid_on(self, canvas, pts):
        tl, tr, br, bl = pts
        n = 5
        for i in range(1, n):
            s = i / n
            p1 = (tl + s * (tr - tl)).astype(int)
            p2 = (bl + s * (br - bl)).astype(int)
            cv2.line(canvas, tuple(p1), tuple(p2), (0, 200, 255), 1)
            p3 = (tl + s * (bl - tl)).astype(int)
            p4 = (tr + s * (br - tr)).astype(int)
            cv2.line(canvas, tuple(p3), tuple(p4), (0, 200, 255), 1)

    def _scanner_pos(self, pts):
        pts_f   = pts.astype(np.float32)
        edges   = [(pts_f[i], pts_f[(i + 1) % 4]) for i in range(4)]
        lengths = [float(np.linalg.norm(b - a)) for a, b in edges]
        perim   = sum(lengths)
        if perim < 1:
            return None
        target, acc = self._scan * perim, 0.0
        for (a, b), L in zip(edges, lengths):
            if acc + L >= target:
                t  = (target - acc) / L if L > 0 else 0.0
                return (int(a[0] + t * (b[0] - a[0])),
                        int(a[1] + t * (b[1] - a[1])))
            acc += L
        return None

    def _draw_readout(self, frame, pts, t):
        if self._alpha < 0.5:
            return
        tl    = pts[0]
        blink = int(t * 2) % 2 == 0
        cv2.putText(frame, "[ REC ]" if blink else "[     ]",
                    (tl[0] + 5, tl[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 100), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ particles

    def _emit(self, corners):
        if len(self._particles) >= self.MAX_PARTICLES:
            return
        if np.random.random() > 0.35:
            return
        ei  = np.random.randint(4)
        s   = np.random.random()
        a, b = corners[ei], corners[(ei + 1) % 4]
        palette = [(0, 255, 255), (0, 180, 255), (100, 255, 200), (180, 120, 255)]
        self._particles.append({
            "x": float(a[0] + s * (b[0] - a[0])),
            "y": float(a[1] + s * (b[1] - a[1])),
            "vx": np.random.uniform(-2.0, 2.0),
            "vy": np.random.uniform(-3.0, -0.5),
            "life":  1.0,
            "decay": np.random.uniform(0.022, 0.055),
            "size":  np.random.randint(2, 5),
            "color": palette[np.random.randint(len(palette))],
        })

    def _age_particles(self):
        alive = []
        for p in self._particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vy"] += 0.06
            p["life"] -= p["decay"]
            if p["life"] > 0:
                alive.append(p)
        self._particles = alive

    def _draw_particles(self, frame):
        pass   # já desenhado dentro do HUD overlay

    # ------------------------------------------------------------------ default image

    @staticmethod
    def _make_default():
        W, H = 800, 600
        img  = np.zeros((H, W, 3), dtype=np.uint8)
        for y in range(H):
            v = y / H
            img[y] = [int(15 + v * 35), int(5 + v * 20), int(40 + v * 80)]
        for x in range(0, W, 40):
            cv2.line(img, (x, 0), (x, H), (0, 70, 110), 1)
        for y in range(0, H, 40):
            cv2.line(img, (0, y), (W, y), (0, 70, 110), 1)
        for i in range(-H, W + H, 80):
            cv2.line(img, (i, 0), (i + H, H), (0, 45, 70), 1)
        for r, c in [(140, (0, 30, 55)), (90, (0, 50, 80)), (50, (0, 70, 110))]:
            cv2.circle(img, (W // 2, H // 2), r, c, -1)
        for r in range(170, 60, -30):
            cv2.circle(img, (W // 2, H // 2), r, (0, 100, 160), 1, cv2.LINE_AA)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "HOLOGRAPHIC", (W // 2 - 195, H // 2 - 25),
                    font, 1.3, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "DISPLAY", (W // 2 - 95, H // 2 + 45),
                    font, 1.3, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "[ SYSTEM ACTIVE ]", (W // 2 - 130, H // 2 + 110),
                    font, 0.65, (0, 140, 200), 1, cv2.LINE_AA)
        for cx, cy in [(50, 50), (W - 50, 50), (W - 50, H - 50), (50, H - 50)]:
            cv2.circle(img, (cx, cy), 22, (0, 140, 200), 1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), 6, (0, 210, 255), -1)
            for angle in (0, 90, 180, 270):
                rad = math.radians(angle)
                ex  = int(cx + 30 * math.cos(rad))
                ey  = int(cy + 30 * math.sin(rad))
                cv2.line(img, (cx, cy), (ex, ey), (0, 140, 200), 1, cv2.LINE_AA)
        return img
