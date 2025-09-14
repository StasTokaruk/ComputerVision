"""
Скрипт демонструє композиційні 2D-перетворення (переміщення, обертання, масштабування)
над правильним шестикутником у гомогенних координатах. Використано матриці 3×3 та їхню
композицію. Предмет: Computer Vision.

Умови завдання:
- Вхідна матриця координат вершин — розширена (гомогенна) 3×N.
- Операції: переміщення (циклічно, траєкторія «прихована»), обертання, масштабування.
- Усі трансформації виконуються в межах графічного вікна.
- Можна змінювати параметри в конфігурації нижче.

Керування під час виконання:
- q — вихід
- p — пауза/продовжити
- r — скинути параметри до початкових

Залежності: numpy, opencv-python
Запуск: python hex_transform_cv.py
"""
import math
from typing import Tuple

import cv2
import numpy as np

# =====================
# Конфігурація сцени
# =====================
WIN_W, WIN_H = 900, 700           # розмір вікна
BG_COLOR = (18, 18, 18)            # фон (B, G, R)
HEX_COLOR = (240, 240, 240)        # колір ребер фігури
CENTER_COLOR = (120, 180, 255)     # колір центру фігури
THICKNESS = 2                      # товщина ліній
MARGIN = 10                        # внутрішній відступ від країв

# Базовий розмір шестикутника (радіус описаного кола)
HEX_R = 90

# Параметри анімації
FPS = 60
DT = 1.0 / FPS

# Обертання (рад/с)
OMEGA = math.radians(28)
# Масштаб (гармонічна модуляція навколо 1.0)
S_BASE = 1.0
S_AMP = 0.25                       # амплітуда (0.25 => від 0.75 до 1.25)
S_FREQ = 0.2                       # Гц

# Переміщення — ЦИКЛІЧНА траєкторія (коло), траєкторія не малюється (лише поточний кадр)
TRANSL_CYC_FREQ = 0.05             # Гц

# =====================
# Допоміжні матричні функції (3×3)
# =====================

def T(tx: float, ty: float) -> np.ndarray:
    """Матриця зсуву.
    [[1, 0, tx],
     [0, 1, ty],
     [0, 0,  1]]
    """
    return np.array([[1.0, 0.0, tx],
                     [0.0, 1.0, ty],
                     [0.0, 0.0, 1.0]], dtype=np.float32)


def R(theta: float) -> np.ndarray:
    """Матриця обертання навколо початку координат (кут у радіанах)."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float32)


def S(sx: float, sy: float) -> np.ndarray:
    """Матриця масштабування відносно початку координат."""
    return np.array([[sx, 0.0, 0.0],
                     [0.0, sy, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float32)


def regular_hexagon(radius: float) -> np.ndarray:
    """Повертає гомогенну матрицю вершин 3×N правильного шестикутника,
    заданого в локальній системі координат із центром у (0,0).
    Порядок вершин — за годинниковою стрілкою.
    """
    pts = []
    for k in range(6):
        ang = math.radians(60 * k)
        x = radius * math.cos(ang)
        y = radius * math.sin(ang)
        pts.append([x, y, 1.0])
    return np.array(pts, dtype=np.float32).T  # форма 3×6


def to_int_points(Mxy1: np.ndarray) -> np.ndarray:
    """Перетворює гомогенні точки 3×N у цілими пікселі Nx1x2 для polylines."""
    # Нормалізація на w (тут w==1)
    pts = (Mxy1[:2, :] / Mxy1[2:3, :]).T  # N×2
    pts = np.round(pts).astype(np.int32)
    return pts.reshape(-1, 1, 2)


def draw_hexagon(canvas: np.ndarray, pts3xN: np.ndarray,
                 color: Tuple[int, int, int], thickness: int = 2) -> None:
    contour = to_int_points(pts3xN)
    cv2.polylines(canvas, [contour], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)


# =====================
# Підготовка сцени
# =====================
hex_local = regular_hexagon(HEX_R)  # 3×6, центр у (0,0)

# Максимальна відстань вершини від центру (для безпечного радіуса траєкторії)
hex_max_r = HEX_R  # для правильного шестикутника це і є радіус описаного кола

# Обмеження для колової траєкторії центру, аби фігура не виходила за межі вікна
max_scale = S_BASE + abs(S_AMP)
safe_radius_x = (WIN_W / 2) - max_scale * hex_max_r - MARGIN
safe_radius_y = (WIN_H / 2) - max_scale * hex_max_r - MARGIN
orbit_R = max(10, min(safe_radius_x, safe_radius_y))  # пікселі

# Початкові стани
angle = 0.0
phase = 0.0  # для масштабування
phi = 0.0    # для колової траєкторії переносу

paused = False

cv2.namedWindow("2D Transforms — Hexagon", cv2.WINDOW_GUI_EXPANDED)
cv2.resizeWindow("2D Transforms — Hexagon", WIN_W, WIN_H)

# =====================
# Головний цикл
# =====================
while True:
    frame = np.full((WIN_H, WIN_W, 3), BG_COLOR, dtype=np.uint8)

    # Центр колової траєкторії — центр екрану
    cx, cy = WIN_W / 2.0, WIN_H / 2.0

    if not paused:
        # 1) Переміщення: ЦИКЛІЧНА траєкторія по колу (x(t), y(t))
        phi += 2 * math.pi * TRANSL_CYC_FREQ * DT
        tx = cx + orbit_R * math.cos(phi)
        ty = cy + orbit_R * math.sin(phi)

        # 2) Обертання: накопичувальний кут
        angle += OMEGA * DT

        # 3) Масштаб: гармонічна модуляція
        phase += 2 * math.pi * S_FREQ * DT
    else:
        tx = cx + orbit_R * math.cos(phi)
        ty = cy + orbit_R * math.sin(phi)

    scale = S_BASE + S_AMP * math.sin(phase)

    # --- Композиція перетворень ---
    # Порядок: спочатку масштаб і обертання в локальному О(0,0), потім перенос у глобальні координати
    # X' = T(tx,ty) * R(angle) * S(scale, scale) * X
    M = T(tx, ty) @ R(angle) @ S(scale, scale)

    # Застосувати до всіх вершин (3×6)
    hex_world = M @ hex_local

    # Малюємо шестикутник
    draw_hexagon(frame, hex_world, HEX_COLOR, THICKNESS)

    # Візуалізація центру фігури (поточне положення, траєкторія НЕ малюється)
    cv2.circle(frame, (int(round(tx)), int(round(ty))), 3, CENTER_COLOR, -1, lineType=cv2.LINE_AA)

    # Рамка вікна для візуального контролю меж
    cv2.rectangle(frame, (MARGIN, MARGIN), (WIN_W - MARGIN, WIN_H - MARGIN), (60, 60, 60), 1, cv2.LINE_AA)

    # Текстова підказка
    info = f"orbitR={int(orbit_R)} | scale={scale:.2f} | angle={math.degrees(angle)%360:6.2f}° | q-quit p-pause r-reset"
    cv2.putText(frame, info, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.imshow("2D Transforms — Hexagon", frame)

    key = cv2.waitKey(int(1000 * DT)) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        paused = not paused
    elif key == ord('r'):
        angle = 0.0
        phase = 0.0
        phi = 0.0
        paused = False

cv2.destroyAllWindows()
