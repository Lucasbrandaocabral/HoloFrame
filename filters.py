"""Suavização de landmarks — Filtro 1€ (One Euro Filter, Casiez et al. 2012).

Pouco lag quando a mão/rosto se move rápido, pouco jitter quando está parado.
Adapta o corte ao módulo da velocidade — ideal para tracking interativo.
"""

import math


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Filtro 1€ escalar, indexado por timestamp real (segundos)."""

    def __init__(self, mincutoff: float = 1.5, beta: float = 0.02, dcutoff: float = 1.0):
        self.mincutoff = mincutoff
        self.beta      = beta
        self.dcutoff   = dcutoff
        self._x  = None
        self._dx = 0.0
        self._t  = None

    def reset(self):
        self._x, self._dx, self._t = None, 0.0, None

    def __call__(self, x: float, t: float) -> float:
        if self._t is None or self._x is None:
            self._t, self._x, self._dx = t, x, 0.0
            return x
        dt = t - self._t
        if dt <= 0.0:
            dt = 1e-3
        self._t = t

        dx     = (x - self._x) / dt
        a_d    = _alpha(self.dcutoff, dt)
        self._dx = a_d * dx + (1.0 - a_d) * self._dx

        cutoff = self.mincutoff + self.beta * abs(self._dx)
        a      = _alpha(cutoff, dt)
        self._x = a * x + (1.0 - a) * self._x
        return self._x


class LandmarkSmoother:
    """Suaviza um dict {idx: (x, y)} com um filtro 1€ por coordenada.

    Mantém o estado por índice de landmark entre chamadas. `reset()` quando o
    alvo some, para a reaquisição entrar limpa (sem pulo a partir de estado velho).
    """

    def __init__(self, mincutoff: float = 1.5, beta: float = 0.02, dcutoff: float = 1.0):
        self._params = (mincutoff, beta, dcutoff)
        self._fx: dict[int, OneEuroFilter] = {}
        self._fy: dict[int, OneEuroFilter] = {}

    def reset(self):
        self._fx.clear()
        self._fy.clear()

    def __call__(self, landmarks: dict, t: float) -> dict:
        out = {}
        for idx, (x, y) in landmarks.items():
            fx = self._fx.get(idx)
            if fx is None:
                fx = OneEuroFilter(*self._params)
                fy = OneEuroFilter(*self._params)
                self._fx[idx], self._fy[idx] = fx, fy
            else:
                fy = self._fy[idx]
            out[idx] = (fx(x, t), fy(y, t))
        return out
