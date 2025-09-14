import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
matplotlib.use("TkAgg")

# Побудова правильного шестикутника
def hexagon(radius=1.0):
    angles = np.linspace(0, 2*np.pi, 7)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.vstack([x, y, np.ones_like(x)])  # розширена матриця

# Матриці перетворень
def translation(tx, ty):
    return np.array([[1, 0, tx],
                     [0, 1, ty],
                     [0, 0, 1]])

def rotation(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle),  np.cos(angle), 0],
                     [0, 0, 1]])

def scaling(sx, sy):
    return np.array([[sx, 0, 0],
                     [0, sy, 0],
                     [0, 0, 1]])

# Базовий шестикутник
hex_coords = hexagon(6)

# Фігура у matplotlib
fig, ax = plt.subplots()
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_aspect('equal')
line, = ax.plot([], [], 'bo-', linewidth=2)

# Функція оновлення кадру
def update(frame):
    # 1. Переміщення
    T = translation(2*np.sin(frame/10), 2*np.cos(frame/10))
    # 2. Обертання
    R = rotation(np.radians(frame))
    # 3. Масштабування
    S = scaling(1 + 0.5*np.sin(frame/20), 1 + 0.5*np.sin(frame/20))

    # Композиція перетворень
    M = T @ R @ S
    transformed = M @ hex_coords

    line.set_data(transformed[0, :], transformed[1, :])
    return line,

ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=True)

plt.show()
