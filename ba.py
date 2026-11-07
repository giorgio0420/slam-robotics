import numpy as np
from geometry import v2t, t2v, wrap_angle


def box_plus_pose(x, dx):
    """x [+] dx = v2t(dx) * v2t(x): perturbazione applicata nel frame mondo."""
    return t2v(v2t(dx) @ v2t(x))


def error_and_jacobian(x, p, z, cam):
    """Residuo di riproiezione e = h(x, p) - z (pixel) e Jacobiani rispetto alla
    perturbazione della posa (dx, dy, dtheta) e del punto (dpx, dpy, dpz).
    Ritorna None se il punto e' (quasi) dietro la camera."""
    c, s = np.cos(x[2]), np.sin(x[2])
    Rt = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])        # Rz(theta)^T
    R_cr = cam["cam_T"][:3, :3].T                            # robot -> camera (solo rotazione)
    t_rc = cam["cam_T"][:3, 3]

    q = p - np.array([x[0], x[1], 0.0])
    p_r = Rt @ q                                             # punto nel frame robot
    X, Y, Z = R_cr @ (p_r - t_rc)                            # punto nel frame camera
    if Z < 0.1:
        return None

    fx, fy, cx, cy = cam["K"][0, 0], cam["K"][1, 1], cam["K"][0, 2], cam["K"][1, 2]
    h = np.array([fx * X / Z + cx, fy * Y / Z + cy])
    e = h - z

    # d(pixel)/d(p_camera)
    J_proj = np.array([[fx / Z, 0, -fx * X / Z ** 2],
                       [0, fy / Z, -fy * Y / Z ** 2]])

    # d(p_robot)/d(dx, dy, dtheta), con x <- v2t(d) * x
    dp_r = np.zeros((3, 3))
    dp_r[:, 0] = -Rt[:, 0]
    dp_r[:, 1] = -Rt[:, 1]
    dp_r[:, 2] = -np.array([-p_r[1], p_r[0], 0.0]) - Rt @ np.array([-x[1], x[0], 0.0])

    J_x = J_proj @ R_cr @ dp_r
    J_p = J_proj @ R_cr @ Rt
    return e, J_x, J_p


def odom_error(xi, xj, Z):
    """Errore del vincolo di odometria tra pose consecutive: Z e' il moto relativo misurato."""
    e = t2v(np.linalg.inv(Z) @ np.linalg.inv(v2t(xi)) @ v2t(xj))
    e[2] = wrap_angle(e[2])
    return e


def odom_error_and_jacobian(xi, xj, Z, eps=1e-6):
    """Come sopra, con Jacobiani 3x3 calcolati per differenze finite (con il boxplus)."""
    e = odom_error(xi, xj, Z)
    J_i = np.zeros((3, 3))
    J_j = np.zeros((3, 3))
    for k in range(3):
        d = np.zeros(3)
        d[k] = eps
        J_i[:, k] = (odom_error(box_plus_pose(xi, d), xj, Z) - e) / eps
        J_j[:, k] = (odom_error(xi, box_plus_pose(xj, d), Z) - e) / eps
    return e, J_i, J_j


def total_error(poses, P, obs, cam, rel_odom, w_pix=1.0, w_odom=1.0):
    """Somma pesata dei ||e||^2: riproiezioni + vincoli di odometria.
    w = 1/sigma, quindi Omega = w^2 * I."""
    chi2 = 0.0
    for i, j, z in obs:
        r = error_and_jacobian(poses[i], P[j], z, cam)
        if r is not None:
            chi2 += w_pix ** 2 * (r[0] @ r[0])
    for i, Z in enumerate(rel_odom):
        e = odom_error(poses[i], poses[i + 1], Z)
        chi2 += w_odom ** 2 * (e @ e)
    return chi2


def bundle_adjustment(poses0, points0, frames, cam, iters=30, sigma_pix=1.0, sigma_odom=1.0):
    """Gauss-Newton (con damping tipo Levenberg-Marquardt) su pose SE(2) e punti 3D.
    La posa 0 resta fissa per eliminare il gauge. Omega = diag(1/sigma^2): sigma_pix e'
    il rumore dei pixel (px), sigma_odom quello del moto relativo tra pose consecutive
    (m e rad). Con i default Omega = I.
    Le sole proiezioni non fissano la scala (monoculare): la fissano i vincoli di odometria
    tra pose consecutive, calcolati dalle pose iniziali poses0."""
    w_pix = 1.0 / sigma_pix
    w_odom = 1.0 / sigma_odom
    rel_odom = [np.linalg.inv(v2t(poses0[i])) @ v2t(poses0[i + 1]) for i in range(len(poses0) - 1)]
    ids = sorted(points0)
    idx = {lid: k for k, lid in enumerate(ids)}
    poses = poses0.copy()
    P = np.array([points0[lid] for lid in ids])
    obs = [(i, idx[lid], pix) for i, fr in enumerate(frames)
           for lid, pix in fr["obs"].items() if lid in idx]

    n_x = 3 * len(poses)
    dim = n_x + 3 * len(P)
    lam = 1e-3
    chi2 = total_error(poses, P, obs, cam, rel_odom, w_pix, w_odom)
    print("iter 0: chi2 =", chi2)

    for it in range(iters):
        H = np.zeros((dim, dim))
        b = np.zeros(dim)
        for i, j, z in obs:
            r = error_and_jacobian(poses[i], P[j], z, cam)
            if r is None:
                continue
            e, J_x, J_p = r[0] * w_pix, r[1] * w_pix, r[2] * w_pix
            a = 3 * i
            c = n_x + 3 * j
            H[a:a + 3, a:a + 3] += J_x.T @ J_x
            H[a:a + 3, c:c + 3] += J_x.T @ J_p
            H[c:c + 3, a:a + 3] += J_p.T @ J_x
            H[c:c + 3, c:c + 3] += J_p.T @ J_p
            b[a:a + 3] += J_x.T @ e
            b[c:c + 3] += J_p.T @ e

        for i, Z in enumerate(rel_odom):
            e, J_i, J_j = odom_error_and_jacobian(poses[i], poses[i + 1], Z)
            e, J_i, J_j = e * w_odom, J_i * w_odom, J_j * w_odom
            a = 3 * i
            c = 3 * (i + 1)
            H[a:a + 3, a:a + 3] += J_i.T @ J_i
            H[a:a + 3, c:c + 3] += J_i.T @ J_j
            H[c:c + 3, a:a + 3] += J_j.T @ J_i
            H[c:c + 3, c:c + 3] += J_j.T @ J_j
            b[a:a + 3] += J_i.T @ e
            b[c:c + 3] += J_j.T @ e

        # posa 0 fissa: tolgo le sue 3 variabili dal sistema
        H = H[3:, 3:]
        b = b[3:]

        improved = False
        for _ in range(10):
            dx = np.zeros(dim)
            dx[3:] = np.linalg.solve(H + lam * np.eye(len(b)), -b)

            new_poses = poses.copy()
            for i in range(len(poses)):
                new_poses[i] = box_plus_pose(poses[i], dx[3 * i:3 * i + 3])
            new_P = P + dx[n_x:].reshape(-1, 3)

            new_chi2 = total_error(new_poses, new_P, obs, cam, rel_odom, w_pix, w_odom)
            if new_chi2 < chi2:
                poses, P, chi2 = new_poses, new_P, new_chi2
                lam *= 0.5
                improved = True
                break
            lam *= 4

        print("iter %d: chi2 = %.2f  lambda = %.2g" % (it + 1, chi2, lam))
        if not improved or np.abs(dx).max() < 1e-6:
            break

    return poses, {lid: P[idx[lid]] for lid in ids}
