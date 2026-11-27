import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

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


def solve_S_costates(phi0, T, k,
                 u, u_x, u_y, u_xx, u_yy, u_xy,
                v, v_x, v_y, v_xx, v_yy, v_xy,
                n_pts=200):

    def func(t, s):
        """s: (x,y,phi,S1,S2,S3,px,py)"""
        x, y, phi = s[:3]
        S = s[3:6]
        px, py = s[6:]
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

        G = np.array([-px * u_x - py * v_x, -px * u_y - py * v_y])

        return np.concatenate([F, G, M @ S])

    c_phi0 = np.cos(phi0)
    s_phi0 = np.sin(phi0)
    # state = (x,y,phi,S1,S2,S3,px,py)
    s0 = [0.0, 0.0, phi0, 0.0, 0.0, 1.0, c_phi0, s_phi0] # state, then S(0) = (0, 0, 1)
    res = solve_ivp(func, (0, T), s0, t_eval=np.linspace(0, T, n_pts),
                    atol=1e-10, rtol=1e-9)
    x, y, phi, S1, S2, S3, px, py = res.y
    phi_T = phi[-1]
    xend, yend = x[-1], y[-1]
    c = 1 / (c_phi0 * (u(xend, yend, T) + k * np.cos(phi_T)) + s_phi0 * (v(xend, yend, T) + k * np.sin(phi_T)))
    px, py = c * px, c * py
    return x, y, phi, S1[-1], S2[-1], px, py # px, py with rho0 = 1

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