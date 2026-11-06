So y approaches 2 asymptotically but never reaches it. Your target has B = 2, exactly on that asymptote. The true minimum time is infinite.

The optimality equation tells the same story. For this flow your dphi_dt reduces to φ' = c·sin φ·cos φ, so tan φ = tan φ₀ · e^{ct}. The heading is pushed to 90° (straight up) exponentially, which is exactly what your right-hand plot shows. Once φ ≈ 90°, x' = cos φ ≈ 0, so x stalls.

What each method is doing
solve_bvp: it settles on T ≈ 49, where y is within the solver tolerance of 2 (2e^{-T/2} is about 1e-11 at T = 49) and x has just crept up to 2. That T is set by tol=1e-8 and the mesh, not by the physics. Tighten the tolerance or change the initial guess and T changes.
Your shooting method: look at your legend. T goes 1.5 → 2.8 → 4.0 → 5.3 → 6.6 → 8.0 → 9.6 → 11.4 → 13.8 → 16.9, and φ₀ climbs toward 90° at the same time. That is Newton chasing T → ∞. It stopped because of num=10, not because it converged. The curves also end around (1.7, 1.97), clearly short of the star, so the loss is still large.

MUST ALWAYS SATISFY

y(t) < k/c for all t