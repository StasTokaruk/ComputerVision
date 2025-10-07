import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
matplotlib.use("TkAgg")

def rotation_matrix(a):
    return np.array([
        [np.cos(a), -np.sin(a), 0],
        [np.sin(a),  np.cos(a), 0],
        [0, 0, 1]
    ])

def scaling_matrix(sx, sy):
    return np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0,  0, 1]
    ])

def translation_matrix(tx, ty):
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])

# Початковий квадрат
square = np.array([
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
    [0, 0, 1]
])
square = square.T

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

line, = ax.plot([], [], 'o-', lw=2, color="blue")

def update(frame):
    angle = np.radians(frame)
    scale = 1 + 0.5 * np.sin(frame * 0.1)
    tx = 2 * np.cos(frame * 0.1)
    ty = 2 * np.sin(frame * 0.1)

    # Композиція перетворень
    transform = translation_matrix(tx, ty) @ rotation_matrix(angle) @ scaling_matrix(scale, scale)
    transformed_square = transform @ square

    line.set_data(transformed_square[0, :], transformed_square[1, :])
    return line,

ani = animation.FuncAnimation(fig, update, frames=360, interval=50, blit=True)
plt.show()
