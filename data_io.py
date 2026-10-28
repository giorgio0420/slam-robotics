import os
import numpy as np


def load_camera(path):
    """Legge camera.dat: K, posa camera nel frame robot (4x4), z_near/z_far, dimensioni immagine."""
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]

    K = np.array([[float(x) for x in lines[i].split()] for i in (1, 2, 3)])
    cam_T = np.array([[float(x) for x in lines[i].split()] for i in (5, 6, 7, 8)])
    z_near = float(lines[9].split(":")[1])
    z_far = float(lines[10].split(":")[1])
    width = int(lines[11].split(":")[1])
    height = int(lines[12].split(":")[1])
    return dict(K=K, cam_T=cam_T, z_near=z_near, z_far=z_far, width=width, height=height)


def load_trajectory(path):
    """Legge trajectoy.dat (il nome del file contiene un refuso nel dataset).
    Ritorna odometria (N,3) e ground truth (N,3), ciascuna come (x, y, theta)."""
    data = np.loadtxt(path)
    return data[:, 1:4], data[:, 4:7]


def load_world(path):
    """Legge world.dat: ritorna dict id -> posizione 3D (serve solo per la valutazione)."""
    data = np.loadtxt(path)
    return {int(row[0]): row[1:4] for row in data}


def load_measurements(data_dir):
    """Legge tutti i meas-XXXXX.dat. Per ogni frame ritorna un dict con
    odom, gt e le osservazioni {landmark_id: (col, row)}."""
    frames = []
    i = 0
    while True:
        path = os.path.join(data_dir, "meas-%05d.dat" % i)
        if not os.path.exists(path):
            break
        odom, gt, obs = None, None, {}
        with open(path) as f:
            for line in f:
                p = line.split()
                if not p:
                    continue
                if p[0] == "gt_pose:":
                    gt = np.array([float(x) for x in p[1:4]])
                elif p[0] == "odom_pose:":
                    odom = np.array([float(x) for x in p[1:4]])
                elif p[0] == "point":
                    obs[int(p[2])] = np.array([float(p[3]), float(p[4])])
        frames.append(dict(odom=odom, gt=gt, obs=obs))
        i += 1
    return frames
