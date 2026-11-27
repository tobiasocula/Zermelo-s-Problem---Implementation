import numpy as np
from core.formulations import Formulation

zeros = lambda t, x, y: np.zeros_like(x, dtype=float)
ones  = lambda t, x, y: np.ones_like(x, dtype=float)

# 1: u = 0, v = -c*y
funcs1 = [
    zeros,                                        # u
    zeros,                                        # u_x
    zeros,                                        # u_y
    zeros,                                        # u_xx
    zeros,                                        # u_yy
    zeros,                                        # u_xy
    lambda t, x, y, c=0.5: -c * y,                # v
    zeros,                                        # v_x
    lambda t, x, y, c=0.5: -c * np.ones_like(y, dtype=float),  # v_y
    zeros,                                        # v_xx
    zeros,                                        # v_yy
    zeros,                                        # v_xy
]
f1 = Formulation(2, 2, 2.0, funcs1)

# 2: u = -y, v = 0
funcs2 = [
    lambda t, x, y: -y,                           # u
    zeros,                                        # u_x
    lambda t, x, y: -np.ones_like(y, dtype=float),  # u_y
    zeros,                                        # u_xx
    zeros,                                        # u_yy
    zeros,                                        # u_xy
    zeros,                                        # v
    zeros,                                        # v_x
    zeros,                                        # v_y
    zeros,                                        # v_xx
    zeros,                                        # v_yy
    zeros,                                        # v_xy
]
f2 = Formulation(2, 2, 1, funcs2)

# 3: sinusoidal cellular flow
funcs3 = [
    lambda t, x, y, c=6, w=np.pi/10: -c * np.sin(w * y),         # u
    zeros,                                                       # u_x
    lambda t, x, y, c=6, w=np.pi/10: -c * w * np.cos(w * y),     # u_y
    zeros,                                                       # u_xx
    lambda t, x, y, c=6, w=np.pi/10:  c * w**2 * np.sin(w * y),  # u_yy
    zeros,                                                       # u_xy
    lambda t, x, y, c=6, w=np.pi/10:  c * np.sin(w * x),         # v
    lambda t, x, y, c=6, w=np.pi/10:  c * w * np.cos(w * x),     # v_x
    zeros,                                                       # v_y
    lambda t, x, y, c=6, w=np.pi/10: -c * w**2 * np.sin(w * x),  # v_xx
    zeros,                                                       # v_yy
    zeros,                                                       # v_xy
]
f3 = Formulation(2, 2, 10, funcs3)

# 4: linear strain u = a*x, v = b*y
funcs4 = [
    lambda t, x, y, a=0.5, b=0.7: a * x,                             # u
    lambda t, x, y, a=0.5, b=0.7: a * np.ones_like(x, dtype=float),  # u_x
    zeros,                                                           # u_y
    zeros,                                                           # u_xx
    zeros,                                                           # u_yy
    zeros,                                                           # u_xy
    lambda t, x, y, a=0.5, b=0.7: b * y,                             # v
    zeros,                                                           # v_x
    lambda t, x, y, a=0.5, b=0.7: b * np.ones_like(y, dtype=float),  # v_y
    zeros,                                                           # v_xx
    zeros,                                                           # v_yy
    zeros,                                                           # v_xy
]
f4 = Formulation(2, 2, 1.0, funcs4)

# 5: u = -y^2, v = 0
funcs5 = [
    lambda t, x, y: -y**2,                                 # u
    zeros,                                                 # u_x
    lambda t, x, y: -2 * y,                                # u_y
    zeros,                                                 # u_xx
    lambda t, x, y: -2 * np.ones_like(y, dtype=float),     # u_yy
    zeros,                                                 # u_xy
    zeros,                                                 # v
    zeros,                                                 # v_x
    zeros,                                                 # v_y
    zeros,                                                 # v_xx
    zeros,                                                 # v_yy
    zeros,                                                 # v_xy
]
f5 = Formulation(2, 2, 1.0, funcs5)

# 6: time-dependent shear u = -t*y, v = 0
funcs6 = [
    lambda t, x, y: -t * y,                                # u
    zeros,                                                 # u_x
    lambda t, x, y: -t * np.ones_like(y, dtype=float),     # u_y
    zeros,                                                 # u_xx
    zeros,                                                 # u_yy
    zeros,                                                 # u_xy
    zeros,                                                 # v
    zeros,                                                 # v_x
    zeros,                                                 # v_y
    zeros,                                                 # v_xx
    zeros,                                                 # v_yy
    zeros,                                                 # v_xy
]
f6 = Formulation(2, 2, 1.0, funcs6)

formulations = [f1, f2, f3, f4, f5, f6]