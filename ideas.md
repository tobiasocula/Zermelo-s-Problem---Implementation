

For stronger verification, you could use:
- Pontryagin's Maximum Principle (PMP): establish the necessary conditions for an optimal trajectory.
- Second variation / conjugate-point analysis: investigate whether an extremal is locally minimizing.
- Hamilton–Jacobi–Bellman (HJB) equation: compute the value function, which provides a route towards global optimality verification.

So, for a candidate numerical trajectory, you can calculate several residuals:
1. Endpoint residual: how close does the trajectory get to \((A,B)\)?
2. State-equation residual: does its numerical derivative agree with the prescribed dynamics?
3. Costate-equation residual: do the costates satisfy their differential equations?
4. Control residual: does the heading minimize the Hamiltonian?
5. Hamiltonian residual: does \(H\) remain close to zero along the trajectory?

