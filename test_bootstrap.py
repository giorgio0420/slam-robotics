import numpy as np
from data_io import load_camera, load_trajectory, load_world, load_measurements
from triangulation import project, triangulate_all

DATA = "data"
cam = load_camera(DATA + "/camera.dat")
odom, gt = load_trajectory(DATA + "/trajectoy.dat")
world = load_world(DATA + "/world.dat")
frames = load_measurements(DATA)

# 1) controllo della proiezione: landmark vero + posa vera -> pixel misurato
errs = []
for i, fr in enumerate(frames):
    for lid, pix in fr["obs"].items():
        uv, ok = project(world[lid], gt[i], cam)
        if uv is not None:
            errs.append(np.linalg.norm(uv - pix))
print("proiezione con GT: n =", len(errs), "| errore pixel medio =", np.mean(errs), "| max =", np.max(errs))


def map_rmse(points):
    e = [np.sum((p - world[lid]) ** 2) for lid, p in points.items()]
    return np.sqrt(np.mean(e))


# 2) triangolazione con pose vere (limite inferiore) e con odometria (bootstrap reale)
p_gt = triangulate_all(frames, gt, cam)
p_od = triangulate_all(frames, odom, cam)
print("triangolati con GT:      ", len(p_gt), "punti | RMSE mappa =", map_rmse(p_gt))
print("triangolati con odometria:", len(p_od), "punti | RMSE mappa =", map_rmse(p_od))
