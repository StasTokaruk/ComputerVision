import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
import math as mt
import random

matplotlib.use("TkAgg")

st = 80  # розмір паралелепіпеда
rotation_speed = 0.5  # швидкість обертання (градусів за кадр)
move_interval = 60  # інтервал переміщення (в кадрах)

Parallelepiped = np.array([
    [0, 0, 0, 1],
    [st, 0, 0, 1],
    [st, st, 0, 1],
    [0, st, 0, 1],
    [0, 0, st, 1],
    [st, 0, st, 1],
    [st, st, st, 1],
    [0, st, st, 1]
])

def TranslationXYZ(Figure, l, m, n):
    f = np.array([
        [1, 0, 0, l],
        [0, 1, 0, m],
        [0, 0, 1, n],
        [0, 0, 0, 1]
    ])
    return Figure.dot(f.T)


def RotateX(Figure, theta_deg):
    theta = mt.radians(theta_deg)
    f = np.array([
        [1, 0, 0, 0],
        [0, mt.cos(theta), -mt.sin(theta), 0],
        [0, mt.sin(theta), mt.cos(theta), 0],
        [0, 0, 0, 1]
    ])
    return Figure.dot(f.T)


def RotateY(Figure, theta_deg):
    theta = mt.radians(theta_deg)
    f = np.array([
        [mt.cos(theta), 0, mt.sin(theta), 0],
        [0, 1, 0, 0],
        [-mt.sin(theta), 0, mt.cos(theta), 0],
        [0, 0, 0, 1]
    ])
    return Figure.dot(f.T)


def RotateZ(Figure, theta_deg):
    theta = mt.radians(theta_deg)
    f = np.array([
        [mt.cos(theta), -mt.sin(theta), 0, 0],
        [mt.sin(theta), mt.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return Figure.dot(f.T)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colors = ["red", "green", "blue", "yellow", "cyan", "orange"]
current_color = random.choice(colors)
current_pos = [random.randint(100, 500), random.randint(100, 500), random.randint(50, 150)]
current_rotation = [0, 0, 0]
parallelepiped = None

ax.set_xlim(0, 600)
ax.set_ylim(0, 600)
ax.set_zlim(0, 600)
ax.set_box_aspect([1, 1, 1])


def update(frame):
    global parallelepiped, current_color, current_pos, current_rotation, colors
    if parallelepiped is not None:
        parallelepiped.remove()

    current_rotation[0] += rotation_speed
    current_rotation[1] += rotation_speed * 0.7
    current_rotation[2] += rotation_speed * 0.3

    if frame % move_interval == 0:
        current_pos[0] = random.randint(100, 500)
        current_pos[1] = random.randint(100, 500)
        current_pos[2] = random.randint(50, 150)
        current_color = random.choice(colors)

    center = np.array([st / 2, st / 2, st / 2, 0])
    centered = Parallelepiped - center

    rotated = RotateX(centered, current_rotation[0])
    rotated = RotateY(rotated, current_rotation[1])
    rotated = RotateZ(rotated, current_rotation[2])

    transformed = rotated + center
    transformed = TranslationXYZ(transformed, current_pos[0], current_pos[1], current_pos[2])

    verts = transformed[:, :3]
    faces = [
        [verts[0], verts[1], verts[2], verts[3]],
        [verts[4], verts[5], verts[6], verts[7]],
        [verts[0], verts[1], verts[5], verts[4]],
        [verts[1], verts[2], verts[6], verts[5]],
        [verts[2], verts[3], verts[7], verts[6]],
        [verts[3], verts[0], verts[4], verts[7]]
    ]

    parallelepiped = Poly3DCollection(faces, facecolors=current_color, edgecolors='black', linewidths=1, alpha=0.9)
    ax.add_collection3d(parallelepiped)

    return []


ani = FuncAnimation(fig, update, frames=1000, interval=20, blit=False)
plt.show()