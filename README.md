<div align="center">

# ◈ HOLO FRAME

**Rastreamento holográfico de mãos em tempo real**

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-FF6F00?style=flat-square&logo=google&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-22C55E?style=flat-square)

*Transforme suas mãos em uma interface holográfica — sem hardware especial, só a webcam.*

</div>

---

## O que é

HoloFrame detecta suas mãos em tempo real e usa os dedos polegar + indicador de cada mão para definir os quatro cantos de uma moldura virtual. Qualquer imagem — ou uma região da sua tela — é mapeada dentro dessa moldura com correção de perspectiva, cercada por efeitos HUD estilo sci-fi.

Mova as mãos. A moldura segue.

---

## Efeitos

| # | Efeito | O que faz |
|---|--------|-----------|
| `1` | **Trilha de Luz** | Rastro neon cyan/magenta seguindo as pontas dos dedos |
| `2` | **Glitch** | Distorção de canal RGB quando a moldura se move rápido |
| `3` | **Rosto Elástico** | Mapeia o rosto e, com a pinça dos dedos, estica ele como borracha (Gomu Gomu) |
| `4` | **Espelho / Tela** | Exibe webcam ou região da tela dentro da moldura |
| `5` | **Tela Virtual** | Captura uma janela do Windows ao vivo e deixa flutuando no espaço com profundidade — agarre com a pinça e posicione (estilo VR) |
| `6` | **Clone Fantasma** | Palma aberta 1 s congela o frame — veja dois de você ao mesmo tempo |
| `7` | **Desenho no Ar** | Indicador esticado desenha neon; paz ✌ troca cor; C limpa |

---

## Controles

```
TAB      Mostrar / ocultar painel de efeitos
L        Ocultar / mostrar esqueleto das mãos
G        Toggle grade de fundo
S        Selecionar região da tela para a moldura
N        Capturar nova tela virtual flutuante (efeito 5)
X        Remover a última tela virtual
+  -  0  Zoom in / out / reset do conteúdo da moldura
R        Recarregar assets/image.png
H        Toggle HUD no feed da câmera virtual
Q        Sair
```

---

## Câmera Virtual

Com `--vcam`, o output vai direto para qualquer app de videochamada como dispositivo de câmera:

```
uv run main.py --vcam
```

Selecione **OBS Virtual Camera** no Discord, Zoom, Teams ou OBS. Requer OBS Studio instalado (registra o driver; não precisa estar aberto).

---

## Stack

- **MediaPipe Tasks API** — detecção de landmarks da mão com modelo `.task` (~25 MB, baixado automaticamente)
- **OpenCV** — captura, warp perspectivo, rendering de HUD e efeitos
- **NumPy** — operações de pixel com buffers pré-alocados para manter 30+ FPS
- **mss** — captura de região de tela em tempo real
- **pyvirtualcam** — envia frames para câmera virtual do sistema

---

## Arquitetura

```
main.py              Orquestração, loop principal, teclado
hand_tracker.py      Wrapper do MediaPipe Tasks (processa em 640×360)
gesture_detector.py  Detecção do gesto de moldura + suavização EMA
image_renderer.py    Warp perspectivo, HUD, partículas
face_tracker.py      Wrapper do MediaPipe FaceLandmarker (478 pontos do rosto)
effects.py           Trilha, Glitch, Rosto Elástico, Espelho, Tela Virtual, Clone, Desenho
ui_panel.py          Painel lateral holográfico in-app
assets/image.png     Imagem exibida na moldura (substitua pela sua)
```

---

<div align="center">

*Feito com OpenCV + MediaPipe*

</div>
