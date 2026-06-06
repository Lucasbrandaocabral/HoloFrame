import numpy as np


class GestureDetector:
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIN_AREA = 8000      # px² — ignore tiny frames
    CONFIRM_FRAMES = 4   # frames before frame is "locked"
    SMOOTH = 0.18        # EMA weight for incoming corners

    def __init__(self):
        self._smoothed = None
        self._confirm = 0
        self.active = False

    # ------------------------------------------------------------------
    def detect(self, hands):
        """Return smoothed (4,2) float32 corners or None."""
        if len(hands) < 2:
            self._reset()
            return None

        # Collect the 4 key points from any two hands
        pts = []
        for hand in hands[:2]:
            lm = hand["landmarks"]
            pts.append(lm[self.THUMB_TIP])
            pts.append(lm[self.INDEX_TIP])

        ordered = self._order(np.array(pts, dtype=np.float32))

        if self._area(ordered) < self.MIN_AREA:
            self._reset()
            return None

        # EMA smoothing
        if self._smoothed is None:
            self._smoothed = ordered.copy()
        else:
            self._smoothed = self.SMOOTH * ordered + (1 - self.SMOOTH) * self._smoothed

        self._confirm += 1
        if self._confirm >= self.CONFIRM_FRAMES:
            self.active = True

        return self._smoothed if self.active else None

    # ------------------------------------------------------------------
    def _reset(self):
        self._confirm = 0
        self.active = False
        # Keep _smoothed so fade-out renderer can still read last position

    @staticmethod
    def _order(pts):
        """Sort 4 points → TL, TR, BR, BL using centroid + angle."""
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
        # arctan2 returns [-π, π]; sort clockwise starting from top-left ≈ -135°
        idx = np.argsort(angles)
        ordered = pts[idx]
        # Rotate so first point is top-left (smallest x+y sum)
        sums = ordered[:, 0] + ordered[:, 1]
        start = np.argmin(sums)
        ordered = np.roll(ordered, -start, axis=0)
        return ordered.astype(np.float32)

    @staticmethod
    def _area(pts):
        """Shoelace formula for quadrilateral area."""
        n = len(pts)
        a = 0.0
        for i in range(n):
            j = (i + 1) % n
            a += pts[i][0] * pts[j][1]
            a -= pts[j][0] * pts[i][1]
        return abs(a) / 2
