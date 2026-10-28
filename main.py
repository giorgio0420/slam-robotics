import numpy as np
from data_io import load_camera, load_trajectory, load_world, load_measurements
from geometry import v2t, t2v

DATA = "data"

cam = load_camera(DATA + "/camera.dat")
odom, gt = load_trajectory(DATA + "/trajectoy.dat")
world = load_world(DATA + "/world.dat")
frames = load_measurements(DATA)

print("K =\n", cam["K"])
print("cam_T =\n", cam["cam_T"])
print("pose:", len(odom), "| frame misure:", len(frames), "| landmark nel mondo:", len(world))

ids = set()
for fr in frames:
    ids.update(fr["obs"].keys())
print("landmark osservati:", len(ids))
print("osservazioni per frame (media):", np.mean([len(fr["obs"]) for fr in frames]))

# controllo: odometria nei meas-* coincide con trajectoy.dat (pose 1..N-1 vs frame 1..N-1)
print("max diff odom meas/traj:", max(np.abs(frames[i]["odom"] - odom[i]).max() for i in range(1, len(frames))))

# controllo v2t/t2v
assert np.allclose(t2v(v2t(odom[10])), odom[10])
