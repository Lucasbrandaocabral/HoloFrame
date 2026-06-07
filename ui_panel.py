import cv2
import numpy as np


class EffectsPanel:
    W      = 230
    ROW_H  = 34
    MARGIN = 12

    def __init__(self, entries: list):
        self.entries = entries
        self.visible = True

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
            return

        h, w    = frame.shape[:2]
        n       = len(self.entries)
        panel_h = 26 + n * self.ROW_H + 8
        x0      = w - self.W - self.MARGIN
        y0      = self.MARGIN

        # Fundo semi-transparente
        ov = frame.copy()
        cv2.rectangle(ov, (x0, y0), (x0 + self.W, y0 + panel_h), (3, 7, 14), -1)
        for i, entry in enumerate(self.entries):
            if getattr(entry[2], entry[3]):
                iy = y0 + 26 + i * self.ROW_H
                cv2.rectangle(ov, (x0 + 1, iy),
                              (x0 + self.W - 1, iy + self.ROW_H - 1),
                              (0, 30, 48), -1)
        cv2.addWeighted(ov, 0.80, frame, 0.20, 0, frame)

        # Borda e divisor
        cv2.rectangle(frame, (x0, y0), (x0 + self.W, y0 + panel_h), (0, 55, 85), 1)
        cv2.line(frame, (x0, y0 + 24), (x0 + self.W, y0 + 24), (0, 40, 62), 1)

        # Título e contador
        active_n = sum(1 for e in self.entries if getattr(e[2], e[3]))
        cv2.putText(frame, "EFEITOS", (x0 + 10, y0 + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 100, 150), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{active_n}/{n}", (x0 + self.W - 30, y0 + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 70, 105), 1, cv2.LINE_AA)

        # Linhas de efeito
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, entry in enumerate(self.entries):
            key_str, label = entry[0], entry[1]
            obj  = entry[2]
            attr = entry[3]
            on   = getattr(obj, attr)
            mid_y = y0 + 26 + i * self.ROW_H + self.ROW_H // 2

            dot_col   = (0, 210, 100) if on else (18, 28, 42)
            num_col   = (0, 150, 210) if on else (40, 52, 68)
            label_col = (180, 192, 210) if on else (55, 68, 85)

            cv2.circle(frame, (x0 + 13, mid_y), 4, dot_col, -1)
            cv2.putText(frame, key_str, (x0 + 24, mid_y + 5),
                        font, 0.36, num_col, 1, cv2.LINE_AA)
            cv2.putText(frame, label, (x0 + 38, mid_y + 5),
                        font, 0.38, label_col, 1, cv2.LINE_AA)

            sub = getattr(obj, "name", None)
            if sub and on and sub not in ("NORMAL", ""):
                cv2.putText(frame, sub, (x0 + self.W - 72, mid_y + 5),
                            font, 0.32, (0, 185, 145), 1, cv2.LINE_AA)
