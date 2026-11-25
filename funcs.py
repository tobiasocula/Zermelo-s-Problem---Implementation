import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def solve_with_S(phi0, T, k,
                 u, u_x, u_y, u_xx, u_yy, u_xy,
                v, v_x, v_y, v_xx, v_yy, v_xy,
                n_pts=200):

    """
    integrate over path whilst keeping S as well (for jacobian of error vector)
    """

    def func(t, s):
        x, y, phi = s[:3]
        S = s[3:]
        sp, cp = np.sin(phi), np.cos(phi)

        ux, uy = u_x(t, x, y), u_y(t, x, y)
        vx, vy = v_x(t, x, y), v_y(t, x, y)

        F = [u(t, x, y) + k*cp,
            v(t, x, y) + k*sp,
            dphi_dt(t, x, y, phi, u_x, u_y, v_x, v_y)]

        dF3_dx   = (v_xx(t, x, y)*sp**2
                    + (u_xx(t, x, y) - v_xy(t, x, y))*sp*cp
                    - u_xy(t, x, y)*cp**2)
        dF3_dy   = (v_xy(t, x, y)*sp**2
                    + (u_xy(t, x, y) - v_yy(t, x, y))*sp*cp
                    - u_yy(t, x, y)*cp**2)
        dF3_dphi = (2*vx*sp*cp
                    + (ux - vy)*(cp**2 - sp**2)
                    + 2*uy*sp*cp)

        M = np.array([[ux,     uy,     -k*sp],
                    [vx,     vy,      k*cp],
                    [dF3_dx, dF3_dy,  dF3_dphi]])

        return np.concatenate([F, M @ S])

    # state = (x,y,phi,S1,S2,S3)
    s0 = [0.0, 0.0, phi0, 0.0, 0.0, 1.0]       # state, then S(0) = (0, 0, 1)
    res = solve_ivp(func, (0, T), s0, t_eval=np.linspace(0, T, n_pts),
                    atol=1e-10, rtol=1e-9)
    x, y, phi, S1, S2, S3 = res.y
    return x, y, phi, S1[-1], S2[-1]

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


def dphi_dt(t, x, y, phi, u_x, u_y, v_x, v_y):
    """from Zermelo papers"""
    return (v_x(t, x, y)*np.sin(phi)**2
            + (u_x(t, x, y) - v_y(t, x, y))*np.sin(phi)*np.cos(phi)
            - u_y(t, x, y)*np.cos(phi)**2)

def solve_one(phi0, T, u, v, u_x, u_y, v_x, v_y, k, n_pts=200):
    """One path integration given initial conditions"""
    x0, y0 = 0, 0
    def func(t, w):
        x, y, phi = w
        return [u(t, x, y) + k*np.cos(phi),
                v(t, x, y) + k*np.sin(phi),
                dphi_dt(t, x, y, phi, u_x, u_y, v_x, v_y)]
    res = solve_ivp(func, (0, T), [x0, y0, phi0],
                    t_eval=np.linspace(0, T, n_pts), atol=1e-8, rtol=1e-8)
    x, y, phi = res.y
    return x, y, phi
    
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
            print('T local:', T_local)
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

class Formulation:
    def __init__(self, A, B, k, funcs):
        self.A = A
        self.B = B
        self.k = k
        self.funcs = funcs

        self.results_shooting = None
        self.results_wind_adj = None
        self.results_branch_wind_adj = None

    def solve_shooting(self, phi_list, T_list, num_iters, num_points):
        self.results_shooting = iterate(phi_list, T_list, self.funcs, self.A, self.B, self.k, num_points, num_iters)
        return self

    def solve_wind_adj(self, stepsize, num_points, num_iters):
        self.results_wind_adj = iterative_wind(
            self.funcs, A=self.A, B=self.B, k=self.k, numpoints=num_points, num=num_iters, stepsize=stepsize
            )
        return self

    def solve_branch_wind_adj(self, stepsize, num_points, num_iters, dphi, num_phi_branches):
        self.results_branch_wind_adj = branch_iterative_wind(
            self.funcs, self.A, self.B, self.k, num_points, num_iters, num_phi_branches, dphi, stepsize
        )
        return self

    def plot_shooting(self):
        assert self.results_shooting is not None
        best_x, best_y = self.results_shooting[:2]
        fig, ax = plt.subplots()
        ax.plot(best_x, best_y)
        ax.scatter(self.A, self.B, color="red")
        ax.scatter(0, 0, color="green")
        plt.show()

    def plot_wind_adj(self):
        assert self.results_wind_adj is not None

        x = self.results_wind_adj[0]
        y = self.results_wind_adj[1]

        fig, ax = plt.subplots()
        for xset, yset in zip(x,y):
            ax.plot(xset, yset)
        ax.scatter(self.A, self.B, color="red")
        ax.scatter(0, 0, color="green")
        plt.show()

    def plot_branch_wind_adj(self):
        assert self.results_branch_wind_adj is not None
        x, y, _, _ = self.results_branch_wind_adj
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.scatter(self.A, self.B, color="red")
        ax.scatter(0, 0, color="green")
        plt.show()

def plot_formulations_together(forms, method):
    """method = shooting, wind_adj, branch_wind_adj"""
    if method == "shooting":
        assert all([f.results_shooting is not None] for f in forms)
        fig, ax = plt.subplots()
        ax.scatter(0, 0, color="green")
        for f in forms:
            best_x, best_y = f.results_shooting[:2]
            ax.plot(best_x, best_y)
            ax.scatter(f.A, f.B, color="red")
    elif method == "wind_adj":
        assert all([f.results_wind_adj is not None] for f in forms)
        fig, ax = plt.subplots()
        ax.scatter(0, 0, color="green")
        for f in forms:
            x, y, _ = f.results_wind_adj
            ax.plot(x[-1], y[-1])
            ax.scatter(f.A, f.B, color="red")
    elif method == "branch_wind_adj":
        assert all([f.results_branch_wind_adj is not None] for f in forms)
        fig, ax = plt.subplots()
        ax.scatter(0, 0, color="green")
        for f in forms:
            x, y, _, _ = f.results_branch_wind_adj
            ax.plot(x, y)
            ax.scatter(f.A, f.B, color="red")
    else:
        raise

    plt.show()