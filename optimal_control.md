general:
x = state (ie. (x,y))
u = control (phi in this case)
p = costate

p has same dimension as x, is adjoint variable that prices the effect of changing the state on the remaining optimal cost.

## Lagrangian

instantaneous cost density. is the integrand of the objective. is a function of the time and control-function.
for this problem: L = 1 (because it is the integrand).

## Control hamiltonian


is a function p(t) in R^n such that H = p^T f(x,u,t) - L(x,u,t) (Pontryagin Hamiltonian equation) where xdot = f(x,u,t)
