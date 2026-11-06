def deepen(num, phi_init, T_init, h_phi0=1e-7, h_T=1e-7, pct_increase_stop=0.001, lmbda=0.5, reached_dist=0.01):
    trajectory = np.empty((num, 2), dtype=object) # x, y
    losses = np.empty(num, dtype=float)
    params = np.empty((num, 2), dtype=object)

    phi, T = phi_init, T_init
    c = 0
    pct = 1.0

    while c < num and pct >= pct_increase_stop:
        x, y = solve_one(phi, T)
        trajectory[c, 0] = x
        trajectory[c, 1] = y
        params[c, 0] = phi
        params[c, 1] = T
        r = np.array([x[-1] - A, y[-1] - B])
        losses[c] = np.linalg.norm(r)

        xp, yp = solve_one(phi + h_phi0, T)
        r_phi = np.array([xp[-1] - A, yp[-1] - B])
        xt, yt = solve_one(phi, T + h_T)
        r_T = np.array([xt[-1] - A, yt[-1] - B])

        J = np.column_stack([(r_phi - r) / h_phi0, (r_T - r) / h_T])
        delta = -np.linalg.pinv(J) @ r            # Newton step in (phi, T)
        phi, T = phi + lmbda * delta[0], max(T + lmbda * delta[1], 1e-3)

        pct = np.linalg.norm(delta) / max(np.linalg.norm([phi, T]), 1e-12)
        c += 1
        print('c:', c)

    reached = np.linalg.norm([x[-1] - A, y[-1] - B]) < reached_dist

    return trajectory, losses, params, c, reached