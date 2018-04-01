import math


## Curve Math
def point_on_curve_1(t, cx, cy, sx, sy, x1, y1, x2, y2):
    ratio = t/100.0
    x3, y3 = multiply_add(sx, sy, x1, y1, ratio)
    x4, y4 = multiply_add(cx, cy, x2, y2, ratio)
    x5, y5 = difference(x3, y3, x4, y4)
    x, y = multiply_add(x3, y3, x5, y5, ratio)
    return x, y


def point_on_curve_2(t, cx, cy, sx, sy, kx, ky, x1, y1, x2, y2, x3, y3):
    ratio = t/100.0
    x4, y4 = multiply_add(sx, sy, x1, y1, ratio)
    x5, y5 = multiply_add(cx, cy, x2, y2, ratio)
    x6, y6 = multiply_add(kx, ky, x3, y3, ratio)
    x1, y1 = difference(x4, y4, x5, y5)
    x2, y2 = difference(x5, y5, x6, y6)
    x4, y4 = multiply_add(x4, y4, x1, y1, ratio)
    x5, y5 = multiply_add(x5, y5, x2, y2, ratio)
    x1, y1 = difference(x4, y4, x5, y5)
    x, y = multiply_add(x4, y4, x1, y1, ratio)
    return x, y


## Ellipse Math
def starting_point_for_ellipse(x, y, rotate):
    # Rotate starting point
    r = math.radians(rotate)
    sin = math.sin(r)
    cos = math.cos(r)
    x, y = rotate_ellipse(x, y, cos, sin)
    return x, y, sin, cos


def point_in_ellipse(x, y, r_sin, r_cos, degree):
    # Find point in ellipse
    r2 = math.radians(degree)
    cos = math.cos(r2)
    sin = math.sin(r2)
    x = x * cos
    y = y * sin
    # Rotate Ellipse
    x, y = rotate_ellipse(y, x, r_sin, r_cos)
    return x, y


def rotate_ellipse(x, y, sin, cos):
    x1, y1 = multiply(x, y, sin)
    x2, y2 = multiply(x, y, cos)
    x = x2 - y1
    y = y2 + x1
    return x, y


## Vector Math
def get_angle(x1, y1, x2, y2):
    dot = dot_product(x1, y1, x2, y2)
    if abs(dot) < 1.0:
        angle = math.acos(dot) * 180/math.pi
    else:
        angle = 0.0
    return angle


def constrain_to_angle(x, y, sx, sy):
    length, nx, ny = length_and_normal(sx, sy, x, y)
    # dot = nx*1 + ny*0 therefore nx
    angle = math.acos(nx) * 180/math.pi
    angle = constraint_angle(angle)
    ax, ay = angle_normal(ny, angle)
    x = sx + ax*length
    y = sy + ay*length
    return x, y


def constraint_angle(angle):
    n = angle//15
    n1 = n*15
    rem = angle - n1
    if rem < 7.5:
        angle = n*15.0
    else:
        angle = (n+1)*15.0
    return angle


def angle_normal(ny, angle):
    if ny < 0.0:
        angle = 360.0 - angle
    radians = math.radians(angle)
    x = math.cos(radians)
    y = math.sin(radians)
    return x, y


def length_and_normal(x1, y1, x2, y2):
    x, y = difference(x1, y1, x2, y2)
    length = vector_length(x, y)
    if length == 0.0:
        x, y = 0.0, 0.0
    else:
        x, y = x/length, y/length
    return length, x, y


def normal(x1, y1, x2, y2):
    junk, x, y = length_and_normal(x1, y1, x2, y2)
    return x, y


def vector_length(x, y):
    length = math.sqrt(x*x + y*y)
    return length


def distance(x1, y1, x2, y2):
    x, y = difference(x1, y1, x2, y2)
    length = vector_length(x, y)
    return length


def dot_product(x1, y1, x2, y2):
    return x1*x2 + y1*y2


def multiply_add(x1, y1, x2, y2, d):
    x3, y3 = multiply(x2, y2, d)
    x, y = add(x1, y1, x3, y3)
    return x, y


def multiply(x, y, d):
    # Multiply vector
    x = x*d
    y = y*d
    return x, y


def add(x1, y1, x2, y2):
    # Add vectors
    x = x1+x2
    y = y1+y2
    return x, y


def difference(x1, y1, x2, y2):
    # Difference in x and y between two points
    x = x2-x1
    y = y2-y1
    return x, y


def midpoint(x1, y1, x2, y2):
    # Midpoint between to points
    x = (x1+x2)/2.0
    y = (y1+y2)/2.0
    return x, y


def perpendicular(x1, y1):
    # Swap x and y, then flip one sign to give vector at 90 degree
    x = -y1
    y = x1
    return x, y
