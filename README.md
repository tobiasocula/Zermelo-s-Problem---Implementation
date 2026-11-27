### Zermelo's Navigation Problem

My Bachelor Thesis for mathematics. Work in progress.

### Current work

On some research into optimal control theory:

https://drive.google.com/file/d/19PkPq9JF9EXWzzlgfpMkUiUzVdlM8ghO/view?usp=sharing

On the solvability of the problem with constant wind $(u,v)$:

https://drive.google.com/file/d/1I8BuXeeir38Wc1vayjyS0G_ySoNGzaIb/view?usp=sharing

### Files

```funcs.py```

The main implementation file for all implementation attemps.

##### solve_with_S

Trajectory integration, with added components in the integration for computing the Jacobian of the error vector

##### deepen

Implementation of the Newton-algorithm for minimizing the error vector coming from the trajectory.

##### dphi_dt

Navigation equation from the original Zermelo's paper. Differential equation involving $\phi$.

##### solve_one

Standard path integration, without the extra variables from solve_with_S.

##### iterate

Performs the deepen-algorithm for every trajectory described from a parameter-grid $(\phi_0,T)$.

##### iterative_wind

Iterative algorithm, where the trajectory's wind strength is gradually increased, ie. $\varepsilon$ is increased from 0 to 1.

##### branch_iterative_wind

Same, but now using a parameter grid only containing the initial steering angle values $\phi_0$.

##### Formulation

Class representing an instance of the Zermelo problem, described by its parameters $A$, $B$, $k$, $u$ and $v$.
This class includes functions for solving the problem by integrating using different algorithms from above, including for plotting.

##### plot_formulations_together

Given Formulation-instances, plot them simultaneously.
