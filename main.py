import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_io import load_camera, load_trajectory, load_world, load_measurements
from triangulation import triangulate_all
from ba import bundle_adjustment
from evaluation import pose_errors, pose_rmse, map_rmse

DATA = "data"
OUT = "output"
os.makedirs(OUT, exist_ok=True)

# --- dati
cam = load_camera(DATA + "/camera.dat")
odom, gt = load_trajectory(DATA + "/trajectoy.dat")
world = load_world(DATA + "/world.dat")
frames = load_measurements(DATA)

# --- bootstrap: triangolazione con le pose di odometria
points0 = triangulate_all(frames, odom, cam)

# --- bundle adjustment
poses, points = bundle_adjustment(odom, points0, frames, cam)

# --- errori
t_od, r_od = pose_rmse(odom, gt)
t_ba, r_ba = pose_rmse(poses, gt)
m_0 = map_rmse(points0, world)
m_ba = map_rmse(points, world)

report = (
    "Punti triangolati: %d\n"
    "                      odometria    dopo BA\n"
    "RMSE traslazione [m]  %9.5f  %9.5f\n"
    "RMSE rotazione [rad]  %9.5f  %9.5f\n"
    "RMSE mappa [m]        %9.5f  %9.5f   (bootstrap -> BA)\n"
    % (len(points), t_od, t_ba, r_od, r_ba, m_0, m_ba)
)
print(report)
with open(OUT + "/results.txt", "w") as f:
    f.write(report)

# --- grafici
plt.figure(figsize=(7, 7))
plt.plot(gt[:, 0], gt[:, 1], "k-", label="ground truth")
plt.plot(odom[:, 0], odom[:, 1], "r--", label="odometria")
plt.plot(poses[:, 0], poses[:, 1], "b-", label="stima BA")
plt.axis("equal")
plt.legend()
plt.title("Traiettoria")
plt.savefig(OUT + "/trajectory.png", dpi=120)

ids = sorted(points)
est = np.array([points[i] for i in ids])
true = np.array([world[i] for i in ids])
plt.figure(figsize=(7, 7))
plt.scatter(true[:, 0], true[:, 1], s=8, c="k", label="ground truth")
plt.scatter(est[:, 0], est[:, 1], s=8, c="b", marker="x", label="stima BA")
plt.axis("equal")
plt.legend()
plt.title("Mappa (vista dall'alto)")
plt.savefig(OUT + "/map.png", dpi=120)

rot, trans = pose_errors(poses, gt)
plt.figure(figsize=(8, 4))
plt.plot(np.linalg.norm(trans, axis=1), label="traslazione [m]")
plt.plot(np.abs(rot), label="rotazione [rad]")
plt.xlabel("coppia di pose")
plt.legend()
plt.title("Errore relativo tra pose consecutive (dopo BA)")
plt.savefig(OUT + "/pose_error.png", dpi=120)
