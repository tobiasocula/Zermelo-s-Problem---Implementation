import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from core.integrating import *

def deepen_H(num, phi_init, T_init, funcs,
           A, B, k,
           lmbda=0.5, reached_dist=0.01,
           pct_increase_stop=0.001,
           num_points=200,
           H_tol=1e-5):

    
    """
    adapted function for performing Newton-algorithm iteratively
    performs H-check: returns error (with details) if final iteration fails Hamiltonian check
    """

    u, u_x, u_y, u_xx, u_yy, u_xy, v, v_x, v_y, v_xx, v_yy, v_xy = funcs


    trajectory = np.empty((num, 2), dtype=object) # x, y
    losses = np.empty(num, dtype=float)
    params = np.empty((num, 2), dtype=object)

    phi, T = phi_init, T_init
    c = 0
    pct = 1.0
    x_final_series = None
    y_final_series = None

    x_last, y_last, phi_last, px_last, py_last = None, None, None, None, None

    while c < num and pct >= pct_increase_stop:
        x, y, phi_seq, dx_dphi, dy_dphi, px, py = solve_S_costates(
            phi, T, k,
            u, u_x, u_y, u_xx, u_yy, u_xy,
            v, v_x, v_y, v_xx, v_yy, v_xy,
            n_pts=num_points
        )
        x_last, y_last, phi_last, px_last, py_last = x, y, phi_seq, px, py
        phi_end = phi_seq[-1]
        r = np.array([x[-1] - A, y[-1] - B])
        losses[c] = np.linalg.norm(r)
        

        """
        for jacobian:
        second column is for T, first column is for phi
        """

        J = np.empty((2, 2), dtype=float)
        J[:,0] = [dx_dphi, dy_dphi]
        J[:,1] = np.array([u(T, x[-1], y[-1]) + k*np.cos(phi_end), v(T, x[-1], y[-1]) + k*np.sin(phi_end)])
        delta = -np.linalg.pinv(J) @ r            # Newton step in (phi, T)
        phi, T = phi + lmbda * delta[0], max(T + lmbda * delta[1], 1e-3)

        pct = np.linalg.norm(delta) / max(np.linalg.norm([phi, T]), 1e-12)
        trajectory[c, 0] = x
        trajectory[c, 1] = y
        x_final_series = x
        y_final_series = y
        params[c, 0] = phi
        params[c, 1] = T
        c += 1

    # H-check
    reached = np.linalg.norm([x[-1] - A, y[-1] - B]) < reached_dist
    if not reached:
        return {"status": "not reached"}

    T_values = np.linspace(0, T, num_points)
    Hmax = np.zeros_like(T_values)
    H = np.zeros_like(T_values)
    for i,(t,phival) in enumerate(zip(T_values, phi_last)):
        uval = u(x_last[t], y_last[t], t)
        vval = v(x_last[t], y_last[t], t)
        pxval, pyval = px[t], py[t]
        Hmax[i] = pxval*uval + pyval*vval + k*np.sqrt(pxval**2 + pyval**2) - 1
        H[i] = pxval*(uval + k*np.cos(phival)) + pyval*(vval + k*np.sin(phival))
    diff = Hmax - H

    failed = np.any(diff > H_tol)
    if failed:
        return {"status": "H_tol failed", "result": diff}

    # last phi = actual phi0
    return {
        "status": "ok",
        "result": [
            trajectory, losses, params, c, phi, T, x_final_series, y_final_series
        ]
    }

def deepen(num, phi_init, T_init, funcs,
           A, B, k,
           lmbda=0.5, reached_dist=0.01,
           pct_increase_stop=0.001,
           num_points=200):

    """
    main function for performing Newton-algorithm iteratively.
    """

    u, u_x, u_y, u_xx, u_yy, u_xy, v, v_x, v_y, v_xx, v_yy, v_xy = funcs


    trajectory = np.empty((num, 2), dtype=object) # x, y
    losses = np.empty(num, dtype=float)
    params = np.empty((num, 2), dtype=object)

    phi, T = phi_init, T_init
    c = 0
    pct = 1.0
    x_final_series = None
    y_final_series = None

    while c < num and pct >= pct_increase_stop:
        x, y, phi_seq, dx_dphi, dy_dphi = solve_with_S(
            phi, T, k,
            u, u_x, u_y, u_xx, u_yy, u_xy,
            v, v_x, v_y, v_xx, v_yy, v_xy,
            n_pts=num_points
        )
        phi_end = phi_seq[-1]
        r = np.array([x[-1] - A, y[-1] - B])
        losses[c] = np.linalg.norm(r)
        

        """
        for jacobian:
        second column is for T, first column is for phi
        """

        J = np.empty((2, 2), dtype=float)
        J[:,0] = [dx_dphi, dy_dphi]
        J[:,1] = np.array([u(T, x[-1], y[-1]) + k*np.cos(phi_end), v(T, x[-1], y[-1]) + k*np.sin(phi_end)])
        delta = -np.linalg.pinv(J) @ r            # Newton step in (phi, T)
        phi, T = phi + lmbda * delta[0], max(T + lmbda * delta[1], 1e-3)

        pct = np.linalg.norm(delta) / max(np.linalg.norm([phi, T]), 1e-12)
        trajectory[c, 0] = x
        trajectory[c, 1] = y
        x_final_series = x
        y_final_series = y
        params[c, 0] = phi
        params[c, 1] = T
        c += 1
        

    reached = np.linalg.norm([x[-1] - A, y[-1] - B]) < reached_dist

    # last phi = actual phi0
    return trajectory, losses, params, c, reached, phi, T, x_final_series, y_final_series



    
def iterate(phi_list, T_list, funcs, A, B, k, num_points, num_iters=20, lmbda=0.5):
    """perform Newtonian step for every pair in the given grid"""
    trajectories = np.empty((len(phi_list), len(T_list)), dtype=object)
    losses_total = np.empty((len(phi_list), len(T_list), num_iters), dtype=float)
    params_total = np.empty((len(phi_list), len(T_list), num_iters, 2), dtype=object)
    num_iters_total = np.empty((len(phi_list), len(T_list)), dtype=int)
    reached_total = np.empty((len(phi_list), len(T_list)), dtype=bool)
    phi0_total = np.empty((len(phi_list), len(T_list)), dtype=float)
    """
    each (i,j) in above:
    trajectories[i,j] = trajectory of shape (num, 2) for x,y (lists)
    losses_total[i,j] = losses of shape (num,) 
    params_total[i,j] = params of shape (num, 2) for phi, T
    num_iters_total[i,j] = number of iterations
    
    """
    for i,phi in enumerate(phi_list):
        for j,T in enumerate(T_list):
            trajectory, losses, params, c, reached, phi0, _, _, _ = deepen(
                num_iters, phi, T, funcs, A=A, B=B, k=k, lmbda=lmbda, num_points=num_points
            )
            trajectories[i,j] = trajectory
            losses_total[i,j] = losses
            params_total[i,j] = params
            num_iters_total[i,j] = c
            reached_total[i,j] = reached
            phi0_total[i,j] = phi0

    if not reached_total.any():
        raise AssertionError("no reached")

    best_x_seq = None
    best_y_seq = None
    best_time = np.inf
    best_phi0 = None
    for i in range(len(phi_list)):
        for j in range(len(T_list)):
            if not reached_total[i,j]:
                continue
            num_iters = num_iters_total[i,j]
            if params_total[i,j][num_iters-1][1] < best_time or best_x_seq is None:
                best_x_seq = trajectories[i,j][num_iters-1, 0]
                best_y_seq = trajectories[i,j][num_iters-1, 1]
                best_time = params_total[i,j][num_iters-1][1]
                best_phi0 = phi0_total[i,j]

    return best_x_seq, best_y_seq, trajectories, losses_total, params_total, num_iters_total, reached_total, best_time, best_phi0

def iterative_wind(funcs, A, B, k, numpoints, num=20, stepsize=0.01):

    """
    algo for stepwise solving from eps = 0 to 1
    """

    all_phi_list = []
    params_total = []
    losses_total = []
    all_x_traj, all_y_traj = [], []
    
    eps_all = np.linspace(0, 1, int(1/stepsize))[1:]
    phi = np.atan2(A, B)
    all_x_traj.append(np.linspace(0, A, numpoints))
    all_y_traj.append(np.linspace(0, B, numpoints))
    all_phi_list.append([phi]*numpoints)
    T = np.hypot(A, B) / k
    for eps in eps_all:
        scaled_funcs = [
            lambda t, x, y, f=f, eps=eps: eps * f(t, x, y)
            for f in funcs
        ]
        tra, losses, params, c, reached, phi, T, final_x, final_y = deepen(num, phi, T, scaled_funcs, A, B, k, num_points=numpoints)
        all_y_traj.append(final_y)
        all_x_traj.append(final_x)
        all_phi_list.append(phi)
        params_total.append(params)
        losses_total.append(losses)

    return all_x_traj, all_y_traj, all_phi_list

def branch_iterative_wind(funcs, A, B, k, numpoints, num_iters, phi_branches_num=5, d_phi=np.pi/10, stepsize=0.01):

    """
    same as iterative wind, but now branching as well
    """

    num_steps = int(1/stepsize)

    trajectories = np.empty((num_steps, phi_branches_num, 2), dtype=object) # x, y
    phis = np.empty((num_steps, phi_branches_num), dtype=float) # all phi0 values
    Ts = np.empty_like(phis)

    Tinit = np.hypot(A, B) / k
    p0 = np.atan2(A, B)
    start_phis = np.linspace(p0 - d_phi, p0 + d_phi, phi_branches_num)

    # initialize arrays
    phis[0, :] = start_phis
    Ts[0, :] = Tinit
    for i in range(phi_branches_num):
        trajectories[0,i,0] = np.linspace(0, A, numpoints)
        trajectories[0,i,1] = np.linspace(0, B, numpoints)

    for j in range(phi_branches_num):
        for i, eps in enumerate(np.linspace(0, 1, num_steps)[1:], 1):
            phi_local = phis[i-1,j]
            T_local = Ts[i-1,j]
            scaled_funcs = [lambda t, x, y, f=f, eps=eps: eps * f(t, x, y) for f in funcs]
            tra, losses, params, c, reached, phi_local, T_local, final_x, final_y = deepen(num_iters, phi_local, T_local, scaled_funcs, A, B, k, num_points=numpoints)
            trajectories[i,j,:] = [final_x, final_y]
            phis[i,j] = phi_local
            Ts[i,j] = T_local

    best_time = np.inf
    best_x, best_y, best_phi0 = None, None, None
    for i in range(phi_branches_num):
        if best_time > Ts[-1, i] or best_x is None:
            best_time = Ts[-1, i]
            best_x = trajectories[-1, i, 0]
            best_y = trajectories[-1, i, 1]
            best_phi0 = phis[-1, i]

    return best_x, best_y, best_time, best_phi0


def brit_wind_ham(funcs, A, B, k, numpoints, num_iters,
                  phi_branches_num=5, d_phi=np.pi/10, stepsize=0.01,
                  H_tol=1e-5):

    """
    adaptive wind, with hamiltonian check
    this attempt: compute H(t) for every path and every t, and disregard paths whose distance from H_max is great enough
    H_tol must hold for every t in order for a path to not get disregarded
    """

    num_steps = int(1/stepsize)

    trajectories = np.empty((num_steps, phi_branches_num, 2), dtype=object) # x, y
    phis = np.empty((num_steps, phi_branches_num), dtype=float) # all phi0 values
    Ts = np.empty_like(phis)

    Tinit = np.hypot(A, B) / k
    p0 = np.atan2(A, B)
    start_phis = np.linspace(p0 - d_phi, p0 + d_phi, phi_branches_num)

    # initialize arrays
    phis[0, :] = start_phis
    Ts[0, :] = Tinit
    for i in range(phi_branches_num):
        trajectories[0,i,0] = np.linspace(0, A, numpoints)
        trajectories[0,i,1] = np.linspace(0, B, numpoints)
    
    for j in range(phi_branches_num):
        for i, eps in enumerate(np.linspace(0, 1, num_steps)[1:], 1):
            phi_local = phis[i-1,j]
            T_local = Ts[i-1,j]
            scaled_funcs = [lambda t, x, y, f=f, eps=eps: eps * f(t, x, y) for f in funcs]
            results = deepen_H(num_iters, phi_local, T_local, scaled_funcs, A, B, k, num_points=numpoints, H_tol=H_tol)
            if results["status"] == "H_tol failed":
                print('FAILED FOR RUN, diff:', results["result"])
                trajectories[i,j,:] = None
                phis[i,j] = None
                Ts[i,j] = None
            if results["status"] == "not reached":
                print('FAILED FOR RUN: NOT REACHED.')
                trajectories[i,j,:] = None
                phis[i,j] = None
                Ts[i,j] = None
            else:
                tra, losses, params, c, phi_local, T_local, final_x, final_y = results["result"]
                trajectories[i,j,:] = [final_x, final_y]
                phis[i,j] = phi_local
                Ts[i,j] = T_local

    best_time = np.inf
    best_x, best_y, best_phi0 = None, None, None
    for i in range(phi_branches_num):
        if best_time > Ts[-1, i] or best_x is None:
            best_time = Ts[-1, i]
            best_x = trajectories[-1, i, 0]
            best_y = trajectories[-1, i, 1]
            best_phi0 = phis[-1, i]

    return best_x, best_y, best_time, best_phi0
