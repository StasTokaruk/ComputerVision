from OpenGL.GL import *
import math as mt


def hollow_cylinder(r1, r2, width, color):
  fragmentation = 100
  glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, color)
  rotation_angle = 2.0 * mt.pi / fragmentation
  half_width = width / 2

  glNormal3f(0.0, 0.0, 1.0)
  __draw_face(r1, r2, rotation_angle, half_width)

  glNormal3f(0.0, 0.0, -1.0)
  __draw_face(r2, r1, rotation_angle, -half_width)

  __draw_outward(r2, rotation_angle, half_width)
  __draw_outward(r1, rotation_angle, -half_width)


def __draw_face(r1, r2, rotation_angle, z):
  glShadeModel(GL_FLAT)
  glBegin(GL_QUAD_STRIP)
  angle = 0
  while angle <= 2.0 * mt.pi:
    glVertex3f(r1 * mt.cos(angle), r1 * mt.sin(angle), z)
    glVertex3f(r2 * mt.cos(angle), r2 * mt.sin(angle), z)
    angle += rotation_angle
  glEnd()


def __draw_outward(r, rotation_angle, z):
  glShadeModel(GL_SMOOTH)
  glBegin(GL_QUAD_STRIP)
  angle = 0
  while angle <= 2.0 * mt.pi:
    glNormal3f(-mt.cos(angle), -mt.sin(angle), 0.0)
    glVertex3f(r * mt.cos(angle), r * mt.sin(angle), z)
    glVertex3f(r * mt.cos(angle), r * mt.sin(angle), -z)
    angle += rotation_angle
  glEnd()
