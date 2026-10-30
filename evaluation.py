import numpy as np
from geometry import v2t, wrap_angle


def pose_errors(poses, gt):
    """Errore sul moto relativo tra pose consecutive (come nel README):
    rel_T = inv(T0)*T1, rel_GT = inv(GT0)*GT1, error_T = inv(rel_T)*rel_GT.
    Ritorna errore di rotazione (rad) e di traslazione (x, y) per ogni coppia."""
    rot = []
    trans = []
    for i in range(len(poses) - 1):
        rel_T = np.linalg.inv(v2t(poses[i])) @ v2t(poses[i + 1])
        rel_GT = np.linalg.inv(v2t(gt[i])) @ v2t(gt[i + 1])
        error_T = np.linalg.inv(rel_T) @ rel_GT
        rot.append(np.arctan2(error_T[1, 0], error_T[0, 0]))
        trans.append(error_T[:2, 2])
    return np.array(rot), np.array(trans)


def pose_rmse(poses, gt):
    """RMSE di traslazione (m) e di rotazione (rad) sul moto relativo."""
    rot, trans = pose_errors(poses, gt)
    return np.sqrt(np.mean(np.sum(trans ** 2, axis=1))), np.sqrt(np.mean(rot ** 2))


def map_rmse(points, world):
    """RMSE tra i punti stimati e quelli veri (solo gli id stimati)."""
    return np.sqrt(np.mean([np.sum((p - world[lid]) ** 2) for lid, p in points.items()]))
