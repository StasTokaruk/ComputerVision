import random

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import sys
from bearing import Bearing, BearingShape

WINDOW_SIZE = (800, 600)
COLORS = {
    'sky': (135 / 255, 206 / 255, 235 / 255, 1.0),
    'metal': (0.75, 0.75, 0.80, 1.0),
    'balls': (1.0, 1.0, 1.0, 1.0)
}

objects_on_line = []


# клас керує станом окремого підшипника на конвеєрі
class InspectionItem:
    def __init__(self, x_start, scale=1.0):
        self.x = x_start
        self.y = 0
        self.rot_y, self.rot_x = 0, 90
        self.scale = scale

        shape = BearingShape(
            radius_1=0.4, radius_2=0.7,
            radius_3=1.1, radius_4=1.4,
            width_1=0.8, width_2=0.5
        )
        # Створення об'єкта
        self.bearing = Bearing((COLORS['metal'], COLORS['balls']), shape_params=shape)
        self.bearing.setup()

    #оновлення фізики: рух та обертання
    def update(self):
        self.x += 0.03
        self.rot_y += 2
        self.rot_x += 2

    #відображення об'єкта
    def draw(self):
        glPushMatrix()

        #переміщення в позицію
        glTranslatef(self.x, self.y, 0)

        #обертання
        glRotatef(self.rot_x, 0, 1, 0)

        #масштабування
        glScalef(self.scale, self.scale, self.scale)

        #налаштування бліків
        glMaterialfv(GL_FRONT, GL_SPECULAR, (1, 1, 1, 1))
        glMaterialf(GL_FRONT, GL_SHININESS, 100.0)

        #виклик малювання геометрії
        self.bearing.draw_geometry()

        glPopMatrix()


#ініціалізація OpenGL
def init():
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_NORMALIZE)
    glDisable(GL_COLOR_MATERIAL)

    glLightfv(GL_LIGHT0, GL_POSITION, (0, 5, 10, 0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.4, 0.4, 0.4, 1))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1))
    glLightfv(GL_LIGHT0, GL_SPECULAR, (1.0, 1.0, 1.0, 1))

    glClearColor(*COLORS['sky'])


#головний цикл малювання кадру
def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    #камера
    gluLookAt(0, 0, 10, 0, 0, 0, 0, 1, 0)

    global objects_on_line
    #логіка спавну підшипників
    if not objects_on_line or objects_on_line[-1].x > -4:
        objects_on_line.append(InspectionItem(x_start=-10, scale=random.uniform(0.8, 1.3)))

    #видаляємо об'єкти, що виїхали за екран
    objects_on_line = [o for o in objects_on_line if o.x < 10]

    for item in objects_on_line:
        item.update()
        item.draw()

    glutSwapBuffers()


def timer(value):
    glutPostRedisplay()
    glutTimerFunc(16, timer, 0)


def reshape(w, h):
    if h == 0: h = 1
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, w / h, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)


if __name__ == '__main__':
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(*WINDOW_SIZE)
    glutCreateWindow(b"Lab9")
    init()
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutTimerFunc(0, timer, 0)
    glutMainLoop()
