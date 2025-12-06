from dataclasses import dataclass
from OpenGL.GL import *
from OpenGL.GLUT import *
from hollow_cylinder import hollow_cylinder
import math


@dataclass
class BearingShape:
    radius_1: float = 0;
    radius_2: float = 0
    radius_3: float = 0;
    radius_4: float = 0
    width_1: float = 0;
    width_2: float = 0


class Bearing:
    def __init__(self, shape_colors, shape_params=BearingShape()):
        self.__shape_colors = shape_colors
        self.__shape_params = shape_params
        self.__obj_display_list = None  # ID для збереженої геометрії

    #компілює геометрію у Display List для швидкого малювання
    def setup(self):
        self.__obj_display_list = glGenLists(1)
        glNewList(self.__obj_display_list, GL_COMPILE)
        self.__draw_components()
        glEndList()

    #викликає вже скомпільований список команд
    def draw_geometry(self):
        if self.__obj_display_list:
            glCallList(self.__obj_display_list)

    def __draw_components(self):
        shape = self.__shape_params
        # Визначаємо кольори
        c1 = self.__shape_colors[0] if self.__shape_colors else (0.6, 0.6, 0.6, 1)
        c2 = self.__shape_colors[1] if len(self.__shape_colors) > 1 else (0.9, 0.9, 0.9, 1)

        #малюємо основні кільця
        rings = [
            (shape.radius_1, shape.radius_2, shape.width_1, c1),  # Внутрішнє
            (shape.radius_3, shape.radius_4, shape.width_1, c1)  # Зовнішнє
        ]
        for r1, r2, w, col in rings:
            hollow_cylinder(r1, r2, w, col)

        #розрахунок параметрів для кульок та бортиків
        gap = shape.radius_3 - shape.radius_2
        ball_radius = gap / 2.0
        mid_radius = shape.radius_2 + ball_radius
        rim_w, rim_h = 0.1, ball_radius * 0.4
        z_offset = (shape.width_1 / 2) - (rim_w / 2)

        #малюємо бортики
        # внутрішнє кільце
        for z_pos in [-z_offset, z_offset]:
            glPushMatrix()
            glTranslatef(0, 0, z_pos)
            hollow_cylinder(shape.radius_2, shape.radius_2 + rim_h, rim_w, c1)
            glPopMatrix()

        # зовнішнє кільце
        for z_pos in [-z_offset, z_offset]:
            glPushMatrix()
            glTranslatef(0, 0, z_pos)
            hollow_cylinder(shape.radius_3 - rim_h, shape.radius_3, rim_w, c1)
            glPopMatrix()

        #малюємо кульки
        num_balls = 12
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, c2)

        for i in range(num_balls):
            angle = (2 * math.pi / num_balls) * i
            x = mid_radius * math.cos(angle)
            y = mid_radius * math.sin(angle)

            glPushMatrix()
            glTranslatef(x, y, 0)
            glutSolidSphere(ball_radius, 20, 20)
            glPopMatrix()
