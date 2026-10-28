import numpy as np


def v2t(v):
    """Pose SE(2) (x, y, theta) -> matrice omogenea 3x3."""
    c, s = np.cos(v[2]), np.sin(v[2])
    return np.array([[c, -s, v[0]],
                     [s,  c, v[1]],
                     [0,  0, 1]])


def t2v(T):
    """Matrice omogenea 3x3 -> (x, y, theta)."""
    return np.array([T[0, 2], T[1, 2], np.arctan2(T[1, 0], T[0, 0])])


def wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))
