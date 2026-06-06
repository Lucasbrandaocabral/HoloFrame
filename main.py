"""
HoloFrame — Rastreamento holográfico de mãos em tempo real.

═══════════════════════════════════════════════════
  USO
───────────────────────────────────────────────────
  uv run main.py              modo normal
  uv run main.py --vcam       ativa câmera virtual
                              (requer OBS instalado)

  CONTROLES DE TECLADO
───────────────────────────────────────────────────
  TAB    mostrar / ocultar painel de efeitos
  Q      sair
  R      recarregar assets/image.png
  G      toggle grade de fundo
  H      toggle HUD na câmera virtual
  +      zoom in   (conteúdo dentro da moldura)
  -      zoom out
  0      resetar zoom

  TECLAS DE EFEITO (ver painel in-app):
  1      Trilha de Luz
  2      Glitch
  3      Screenshot automático (segure 2 s parado)
  4      Espelho — webcam dentro da moldura
  S      Selecionar região da tela para usar na moldura
  5      Filtro de Cor (cicla entre modos)
  6      Explosão — feche o punho dentro da moldura
═══════════════════════════════════════════════════
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

from effects import (
    ColorFilter, ExplosionEffect, GlitchEffect,
    LightTrail, MirrorMode, ScreenshotCapture,
)
from gesture_detector import GestureDetector
from hand_tracker import HandTracker
from image_renderer import ImageRenderer
from ui_panel import EffectsPanel

ASSETS_DIR    = os.path.join(os.path.dirname(__file__), "assets")
DEFAULT_IMAGE = os.path.join(ASSETS_DIR, "image.png")
MEDIAPIPE_EVERY_N = 2


# ──────────────────────────────────────────────────────────────────────────────

def _build_grid_cache(h: int, w: int) -> np.ndarray:
    layer = np.zeros((h, w, 3), dtype=np.uint8)
    for x in range(0, w, 60):
        cv2.line(layer, (x, 0), (x, h), (0, 14, 22), 1)
    for y in range(0, h, 60):
        cv2.line(layer, (0, y), (w, y), (0, 14, 22), 1)
    return layer


def _draw_top_hud(frame: np.ndarray, fps: float, hand_count: int, frame_active: bool):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Barra superior
    cv2.rectangle(frame, (0, 0), (w, 38), (0, 12, 22), -1)
    cv2.line(frame, (0, 38), (w, 38), (0, 90, 140), 1)

    cv2.putText(frame, "HOLO FRAME", (12, 26), font, 0.68, (0, 210, 255), 1, cv2.LINE_AA)

    # FPS
    fps_col = (0, 255, 100) if fps >= 28 else (0, 180, 255) if fps >= 15 else (0, 80, 255)
    cv2.putText(frame, f"FPS {fps:4.1f}", (w - 130, 26), font, 0.58, fps_col, 1, cv2.LINE_AA)

    # Ícones de mãos
    for i in range(2):
        col = (0, 255, 150) if i < hand_count else (0, 45, 70)
        cx = w // 2 - 18 + i * 36
        cv2.circle(frame, (cx, 20), 8, col, -1)
        cv2.circle(frame, (cx, 20), 9, (0, 190, 255), 1, cv2.LINE_AA)

    # Status na barra inferior
    if frame_active:
        status, col = "FRAME LOCKED",      (0, 255, 150)
    elif hand_count == 2:
        status, col = "ALINHANDO...",       (0, 200, 255)
    elif hand_count == 1:
        status, col = "UMA MÃO DETECTADA", (0, 150, 255)
    else:
        status, col = "MOSTRE AMBAS AS MÃOS", (55, 75, 110)

    cv2.putText(frame, status, (12, h - 12), font, 0.52, col, 1, cv2.LINE_AA)

    # Dica de gesto
    cv2.putText(frame, "Polegar + indicador = cantos da moldura",
                (12, h - 30), font, 0.36, (38, 68, 90), 1, cv2.LINE_AA)

    # Linha de scan de fundo
    scan_y = int((time.time() * 55) % h)
    frame[scan_y] = np.clip(frame[scan_y].astype(np.int32) + 16, 0, 255)


# ──────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="HoloFrame")
    p.add_argument("--vcam", action="store_true",
                   help="Envia o vídeo para uma câmera virtual (requer OBS instalado)")
    p.add_argument("--vcam-fps", type=int, default=30,
                   help="FPS da câmera virtual (padrão: 30)")
    return p.parse_args()


def _init_vcam(w: int, h: int, fps: int):
    """Tenta criar câmera virtual. Retorna o objeto ou None."""
    try:
        import pyvirtualcam
        cam = pyvirtualcam.Camera(width=w, height=h, fps=fps,
                                  fmt=pyvirtualcam.PixelFormat.BGR,
                                  print_fps=False)
        print(f"Câmera virtual ativa → {cam.device}")
        print("Selecione esse dispositivo no Discord / Zoom / OBS etc.")
        return cam
    except ImportError:
        print("AVISO: pyvirtualcam não instalado.")
        print("  Execute:  uv add pyvirtualcam")
        return None
    except Exception as e:
        print(f"AVISO: câmera virtual indisponível — {e}")
        print("  Verifique se o OBS Studio está instalado.")
        return None


def main():
    args = _parse_args()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERRO: Webcam não encontrada.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ── Módulos principais ────────────────────────────────────────────────
    tracker  = HandTracker()
    detector = GestureDetector()
    renderer = ImageRenderer(DEFAULT_IMAGE if os.path.exists(DEFAULT_IMAGE) else None)

    # ── Efeitos ───────────────────────────────────────────────────────────
    trail      = LightTrail()
    glitch     = GlitchEffect()
    screenshot = ScreenshotCapture()
    mirror     = MirrorMode()
    color_filt = ColorFilter()
    explosion  = ExplosionEffect()

    # ── Painel UI ─────────────────────────────────────────────────────────
    panel = EffectsPanel([
        ("1", "TRILHA DE LUZ",   trail,      "enabled"),
        ("2", "GLITCH",          glitch,     "enabled"),
        ("3", "SCREENSHOT",      screenshot, "enabled"),
        ("4", "ESPELHO",         mirror,     "enabled"),
        ("5", "FILTRO DE COR",   color_filt, "enabled"),
        ("6", "EXPLOSÃO",        explosion,  "enabled"),
    ])

    # ── Câmera virtual (opcional) ─────────────────────────────────────────
    vcam     = _init_vcam(cam_w, cam_h, args.vcam_fps) if args.vcam else None
    vcam_hud = False   # H: inclui HUD no feed da câmera virtual

    # ── Estado global ─────────────────────────────────────────────────────
    grid_cache      = _build_grid_cache(cam_h, cam_w)
    show_grid       = True
    show_landmarks  = True
    zoom_level      = 1.0
    prev_frame: np.ndarray | None = None

    fps_times: list[float] = []
    hands   = []
    corners = None
    frame_idx = 0

    print(__doc__)
    print("Iniciando… pressione TAB para o painel de efeitos.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        # ── MediaPipe (a cada N frames) ───────────────────────────────────
        if frame_idx % MEDIAPIPE_EVERY_N == 0:
            tracker.process(frame)
            hands   = tracker.get_hands(frame)
            corners = detector.detect(hands)
            renderer.set_active(corners is not None)

        frame_idx += 1

        # ── Fundo ─────────────────────────────────────────────────────────
        if show_grid:
            cv2.add(frame, grid_cache, dst=frame)

        # ── Atualiza efeitos ──────────────────────────────────────────────
        trail.update(hands)
        glitch.update(corners)
        explosion.update(hands, corners)

        # ── Trilha de luz ─────────────────────────────────────────────────
        if trail.enabled:
            trail.draw(frame)

        # ── Configura renderer para este frame ────────────────────────────
        renderer.source_frame  = mirror.get_frame(prev_frame)
        renderer.zoom_factor   = zoom_level
        renderer.glitch_active = glitch.is_active
        renderer.color_filter  = color_filt if color_filt.enabled else None

        # ── Landmarks ─────────────────────────────────────────────────────
        if show_landmarks:
            tracker.draw_landmarks(frame, hands)

        # ── Renderiza imagem holográfica ──────────────────────────────────
        frame = renderer.render(frame, corners)

        # ── Screenshot ────────────────────────────────────────────────────
        if screenshot.enabled:
            screenshot.update(frame, corners)
            screenshot.draw(frame, corners)

        # ── Explosão ──────────────────────────────────────────────────────
        if explosion.enabled:
            explosion.draw(frame)

        # ── Salva frame para webcam mirror do próximo ciclo ──────────────
        if mirror.enabled and not mirror.screen_mode:
            prev_frame = frame.copy()
        elif not mirror.enabled:
            prev_frame = None

        # ── Frame limpo para câmera virtual (sem HUD/painel) ─────────────
        vcam_frame = frame.copy() if (vcam is not None and not vcam_hud) else None

        # ── HUD + Painel ──────────────────────────────────────────────────
        now = time.time()
        fps_times.append(now)
        fps_times = [t for t in fps_times if now - t < 1.0]
        fps = float(len(fps_times))

        _draw_top_hud(frame, fps, len(hands), detector.active)
        panel.draw(frame)

        # Indicador de câmera virtual na tela
        if vcam is not None:
            h_fr, w_fr = frame.shape[:2]
            dot_col = (0, 255, 80) if not vcam_hud else (0, 200, 255)
            cv2.circle(frame, (w_fr - 18, 20), 6, dot_col, -1)
            cv2.putText(frame, "VCAM", (w_fr - 70, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, dot_col, 1, cv2.LINE_AA)

        cv2.imshow("HoloFrame", frame)

        # ── Envia para câmera virtual ─────────────────────────────────────
        if vcam is not None:
            out = frame if vcam_hud else vcam_frame
            try:
                vcam.send(out)
            except Exception:
                pass   # não travar o app se a cam virtual desconectar

        # ── Teclado ───────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            renderer.load_image(DEFAULT_IMAGE if os.path.exists(DEFAULT_IMAGE) else None)
            print("Imagem recarregada.")
        elif key == ord("g"):
            show_grid = not show_grid
        elif key == ord("h") and vcam is not None:
            vcam_hud = not vcam_hud
            print("HUD na câmera virtual:", "ON" if vcam_hud else "OFF")
        elif key == ord("s"):
            mirror.select_region()
        elif key == ord("l"):
            show_landmarks = not show_landmarks
        elif key == 9:          # TAB
            panel.toggle()
        elif key == ord("+") or key == ord("="):
            zoom_level = min(4.0, zoom_level + 0.1)
        elif key == ord("-"):
            zoom_level = max(0.25, zoom_level - 0.1)
        elif key == ord("0"):
            zoom_level = 1.0
        else:
            panel.handle_key(key)

    if vcam is not None:
        vcam.close()
    cap.release()
    cv2.destroyAllWindows()
    tracker.release()
    print("Encerrado.")


if __name__ == "__main__":
    main()
