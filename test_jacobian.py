import numpy as np
from data_io import load_camera
from ba import box_plus_pose, error_and_jacobian

cam = load_camera("data/camera.dat")
x = np.array([1.0, -0.5, 0.7])
p = np.array([3.0, 0.5, 0.4])
z = np.array([300.0, 250.0])

e, J_x, J_p = error_and_jacobian(x, p, z, cam)
eps = 1e-6

Jx_num = np.zeros((2, 3))
for k in range(3):
    d = np.zeros(3)
    d[k] = eps
    e2, _, _ = error_and_jacobian(box_plus_pose(x, d), p, z, cam)
    Jx_num[:, k] = (e2 - e) / eps

Jp_num = np.zeros((2, 3))
for k in range(3):
    d = np.zeros(3)
    d[k] = eps
    e2, _, _ = error_and_jacobian(x, p + d, z, cam)
    Jp_num[:, k] = (e2 - e) / eps

print("max |J_x - numerico| =", np.abs(J_x - Jx_num).max(), "(scala J:", np.abs(J_x).max(), ")")
print("max |J_p - numerico| =", np.abs(J_p - Jp_num).max(), "(scala J:", np.abs(J_p).max(), ")")
