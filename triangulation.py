import numpy as np


def robot_to_world(pose):
    """Posa SE(2) (x, y, theta) -> matrice 4x4 robot->mondo (il robot sta sul piano z=0)."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return np.array([[c, -s, 0, pose[0]],
                     [s,  c, 0, pose[1]],
                     [0,  0, 1, 0],
                     [0,  0, 0, 1]])


def camera_to_world(pose, cam_T):
    """Posa della camera nel mondo = posa robot * posa camera nel robot."""
    return robot_to_world(pose) @ cam_T


def project(p_world, pose, cam):
    """Proietta un punto 3D del mondo nell'immagine. Ritorna (pixel [col,row], ok).
    ok = False se il punto e' dietro la camera o fuori da z_near/z_far/immagine."""
    T_cw = np.linalg.inv(camera_to_world(pose, cam["cam_T"]))
    p_c = T_cw[:3, :3] @ p_world + T_cw[:3, 3]
    if p_c[2] <= cam["z_near"] or p_c[2] > cam["z_far"]:
        return None, False
    uv = (cam["K"] @ p_c)[:2] / p_c[2]
    ok = 0 <= uv[0] < cam["width"] and 0 <= uv[1] < cam["height"]
    return uv, ok


def pixel_ray(pixel, pose, cam):
    """Raggio nel mondo che passa per il pixel: (origine camera, direzione unitaria)."""
    T_wc = camera_to_world(pose, cam["cam_T"])
    d_c = np.linalg.inv(cam["K"]) @ np.array([pixel[0], pixel[1], 1.0])
    d_w = T_wc[:3, :3] @ d_c
    return T_wc[:3, 3], d_w / np.linalg.norm(d_w)


def triangulate_point(rays):
    """Punto che minimizza la somma dei quadrati delle distanze dai raggi (o_i, d_i).
    Distanza di p dal raggio: (I - d d^T)(p - o).  Sommando: A p = b."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d in rays:
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ o
    p = np.linalg.solve(A, b)
    # il punto deve stare davanti a tutte le camere
    for o, d in rays:
        if (p - o) @ d <= 0:
            return None
    return p


def triangulate_all(frames, poses, cam, min_obs=2, min_cond=1e-3):
    """Bootstrap: per ogni landmark raccoglie i raggi da tutti i frame in cui compare
    (poses = pose usate per i raggi, es. odometria) e li triangola insieme.
    Scarta i landmark con poche viste o con raggi quasi paralleli (A mal condizionata)."""
    rays = {}
    for i, fr in enumerate(frames):
        for lid, pix in fr["obs"].items():
            rays.setdefault(lid, []).append(pixel_ray(pix, poses[i], cam))

    points = {}
    for lid, r in rays.items():
        if len(r) < min_obs:
            continue
        A = sum(np.eye(3) - np.outer(d, d) for _, d in r)
        w = np.linalg.eigvalsh(A)
        if w[0] < min_cond * len(r):
            continue
        p = triangulate_point(r)
        if p is not None:
            points[lid] = p
    return points
