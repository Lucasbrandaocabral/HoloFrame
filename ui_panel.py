"""
Painel lateral holográfico de efeitos.

Cada entrada da lista `entries` é uma tupla:
    (tecla: str, label: str, objeto_efeito, atributo: str)

Se o objeto tiver um método `cycle()`, pressionar a tecla chama cycle()
em vez de alternar `enabled`.
"""

import time

import cv2
import numpy as np


class EffectsPanel:
    W       = 248   # largura do painel em px
    ROW_H   = 38    # altura de cada linha de efeito
    MARGIN  = 12    # margem direita/topo

    def __init__(self, entries: list):
        self.entries = entries
        self.visible = True
        self._t0     = time.time()

    # ------------------------------------------------------------------ public

    def toggle(self):
        self.visible = not self.visible

    def handle_key(self, key: int) -> bool:
        for entry in self.entries:
            if key == ord(entry[0]):
                obj = entry[2]
                if callable(getattr(obj, "cycle", None)):
                    obj.cycle()
                else:
                    attr = entry[3]
                    setattr(obj, attr, not getattr(obj, attr))
                return True
        return False

    def draw(self, frame: np.ndarray):
        if not self.visible:
            # Mostrar hint minimalista
            h, w = frame.shape[:2]
            cv2.putText(frame, "[TAB] efeitos", (w - 135, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 80, 100), 1, cv2.LINE_AA)
            return

        h, w    = frame.shape[:2]
        t       = time.time() - self._t0
        n       = len(self.entries)
        panel_h = 38 + n * self.ROW_H + 22
        x0      = w  - self.W - self.MARGIN
        y0      = self.MARGIN + 40   # abaixo da barra de HUD

        # ── Fundo + destaques de linhas ativas — 1 único overlay ──────────
        ov = frame.copy()
        cv2.rectangle(ov, (x0, y0), (x0 + self.W, y0 + panel_h), (0, 10, 20), -1)
        for i, entry in enumerate(self.entries):
            if getattr(entry[2], entry[3]):
                iy = y0 + 38 + i * self.ROW_H
                cv2.rectangle(ov, (x0 + 1, iy),
                              (x0 + self.W - 1, iy + self.ROW_H - 2),
                              (0, 45, 65), -1)
        cv2.addWeighted(ov, 0.84, frame, 0.16, 0, frame)

        # ── Bordas e cantos ───────────────────────────────────────────────
        cv2.rectangle(frame, (x0, y0), (x0 + self.W, y0 + panel_h), (0, 120, 170), 1)
        self._corners(frame, x0, y0, x0 + self.W, y0 + panel_h)

        # Divisor
        cv2.line(frame, (x0, y0 + 32), (x0 + self.W, y0 + 32), (0, 60, 100), 1)

        # Título
        cv2.putText(frame, "EFFECTS", (x0 + 10, y0 + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 210, 255), 1, cv2.LINE_AA)

        # Indicador de contagem de efeitos ativos
        active_n = sum(1 for e in self.entries if getattr(e[2], e[3]))
        col_n = (0, 255, 150) if active_n else (60, 60, 80)
        cv2.putText(frame, f"{active_n}/{n}", (x0 + self.W - 40, y0 + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, col_n, 1, cv2.LINE_AA)

        # ── Linhas de efeito ──────────────────────────────────────────────
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, entry in enumerate(self.entries):
            key_str, label = entry[0], entry[1]
            obj  = entry[2]
            attr = entry[3]
            on   = getattr(obj, attr)
            iy   = y0 + 38 + i * self.ROW_H
            mid_y = iy + self.ROW_H // 2

            # Dot de status
            dot_col  = (0, 255, 140) if on else (30, 40, 50)
            ring_col = (0, 180, 100) if on else (45, 50, 60)
            cv2.circle(frame, (x0 + 16, mid_y), 6, dot_col, -1)
            cv2.circle(frame, (x0 + 16, mid_y), 8, ring_col, 1, cv2.LINE_AA)
            if on:
                pulse = int(10 + abs(np.sin(t * 2.6 + i * 1.1)) * 5)
                cv2.circle(frame, (x0 + 16, mid_y), pulse, (0, 160, 90), 1, cv2.LINE_AA)

            # Badge da tecla
            badge_col = (0, 180, 255) if on else (38, 50, 65)
            cv2.rectangle(frame, (x0 + 28, mid_y - 9), (x0 + 44, mid_y + 9), badge_col, 1)
            cv2.putText(frame, key_str.upper(), (x0 + 31, mid_y + 5),
                        font, 0.37, badge_col, 1, cv2.LINE_AA)

            # Label
            label_col = (210, 220, 255) if on else (85, 95, 110)
            cv2.putText(frame, label, (x0 + 51, mid_y + 5),
                        font, 0.44, label_col, 1, cv2.LINE_AA)

            # Submensagem — nome do filtro atual, zoom, etc.
            sub = getattr(obj, "name", None)
            if sub and on and sub != "NORMAL":
                cv2.putText(frame, sub, (x0 + self.W - 95, mid_y + 5),
                            font, 0.38, (0, 240, 200), 1, cv2.LINE_AA)

        # ── Rodapé ────────────────────────────────────────────────────────
        footer_y = y0 + panel_h - 7
        cv2.putText(frame, "TAB  mostrar / ocultar",
                    (x0 + 10, footer_y),
                    font, 0.33, (38, 72, 95), 1, cv2.LINE_AA)

        # Hint de zoom
        cv2.putText(frame, "+ / -  zoom     0  reset",
                    (x0 + 10, footer_y - 14),
                    font, 0.33, (38, 72, 95), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _corners(frame, x0, y0, x1, y1, length=16, color=(0, 210, 255), thick=2):
        for cx, cy, dx, dy in [(x0, y0, 1, 1), (x1, y0, -1, 1),
                                (x1, y1, -1, -1), (x0, y1, 1, -1)]:
            cv2.line(frame, (cx, cy), (cx + dx * length, cy), color, thick)
            cv2.line(frame, (cx, cy), (cx, cy + dy * length), color, thick)
