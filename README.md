# Newton-Raphson load flow solver

A power flow solver written from scratch in C++17: assemble the bus admittance
matrix, derive the power-flow Jacobian by hand, and solve for the steady-state
voltage at every bus with Newton-Raphson. No power-systems library is involved —
the only dependency is Eigen, and only for the linear solve.

It converges the IEEE 14-bus test system in **4 iterations**, matching the
published solution to 5e-4 p.u. — the resolution the reference is quoted to.

## Key features

- **Y-bus assembly by stamping**, so parallel branches, off-nominal transformer
  taps, line charging and bus shunts all compose instead of special-casing each
  other.
- **An analytic Jacobian**, derived by hand rather than approximated by finite
  differences — which is what gives the method its quadratic convergence.
- **Full bus-type handling**: slack, PV and PQ, with the unknown vector sized to
  the buses that actually have unknowns.
- **Per-branch flow and loss reporting**, computed from the same admittance
  function the matrix was built from.
- **SVG output** for the voltage profile and the convergence history, so the
  quadratic error decay is visible rather than asserted.
- **Self-contained**: one translation unit, one compiler invocation, no build
  system.

## How it works

**Y-bus.** Each branch stamps a 2×2 admittance into a complex matrix: `−y/a`
off-diagonal and `y + jb/2` on the diagonal, with the tap-side diagonal scaled
by `1/a²` for transformers. Bus shunts add `Gs + jBs`. Stamps accumulate, so
parallel branches add rather than overwrite.

Stamping is the choice that keeps the model composable. Writing the matrix
entries directly would mean every combination — two lines in parallel, a tapped
transformer with charging, a shunt at a tapped bus — needing its own case.
Stamping makes each element responsible only for its own contribution, and the
matrix falls out of the sum.

**Newton-Raphson.** Power is quadratic in voltage, so

```
P_i = V_i Σ_k V_k (G cos θ_ik + B sin θ_ik)
```

has no closed form. The solver linearizes with the Jacobian

```
J = [ ∂P/∂θ   ∂P/∂|V| ]
    [ ∂Q/∂θ   ∂Q/∂|V| ]
```

solves `J Δx = mismatch` with Eigen's LU, and iterates from a flat start.
Unknowns are θ at every non-slack bus and |V| at every PQ bus — 22 unknowns for
the 14-bus system.

**The diagonal Jacobian terms collapse into the already-computed injections**
(`∂P_i/∂θ_i = −Q_i − B_ii V_i²` and friends). That identity is worth using for
two reasons: it is cheaper, since the injections are needed for the mismatch
anyway, and it is far easier to get right than differentiating the sum
directly — which matters, because a wrong partial derivative does not produce a
wrong answer, it produces a *slowly converging correct* answer, and that is much
harder to notice.

**Why Newton-Raphson at all.** Gauss-Seidel is easier to write and converges
linearly; fast-decoupled is faster per iteration but relies on the `R << X`
approximation that distribution networks violate. Full Newton costs a Jacobian
factorisation per iteration and buys quadratic convergence and no assumption
about the network's X/R ratio. At transmission scale with a sparse solver it is
the standard choice, and at 14 buses it converges in four iterations from a flat
start.

**Branch flows** reuse the same `branch_admittance()` the matrix was built from,
so the flow report cannot drift out of sync with the model the solver used.
Losses come out as the sum of the two directional flows on each branch, which
makes the per-branch loss an independent quantity rather than a restatement of
the solve.

On a log axis the convergence plot bends steadily downward rather than running
straight: each Newton step roughly squares the previous error.

## Scope

Balanced positive-sequence steady state. Deliberately not included: optimal
power flow, dynamics and stability, three-phase unbalanced modelling, and
generator reactive limits.

Dense matrices throughout. Real grids are ~99 % sparse and production solvers
exploit that, but at 14 buses sparsity buys nothing and costs a dependency —
the Jacobian is 22 × 22, and dense LU on that is not measurable.

## Possible improvements

- **Generator reactive limits (PV → PQ switching).** The largest single gap.
  Real generators run out of vars, and when one does, its bus stops holding
  voltage and becomes a PQ bus at its limit. Implementing it means re-typing
  buses between iterations and resizing the unknown vector, which also means
  handling the oscillation where a bus flips back and forth between types.
- **Sparse storage and an ordered sparse factorisation.** The step that makes
  the solver scale past a few hundred buses. Approximate minimum degree ordering
  plus a sparse LU is the textbook route, and Eigen already has both — the work
  is in the assembly, not the solve.
- **Fast-decoupled load flow as a second method.** Cheap to add once the Jacobian
  exists, and having both lets the solver demonstrate the trade directly: FDLF
  takes more iterations but skips the refactorisation, and it degrades exactly
  where the `R << X` assumption fails.
- **Common data formats.** Reading IEEE Common Data Format or MATPOWER cases
  would open every published test system at once, instead of the two networks
  currently compiled in.
- **Tap-changing transformers and phase shifters under control**, rather than
  fixed taps — the discrete-control version of the PV/PQ switching problem.
- **Better flat-start robustness.** Newton diverges on heavily loaded or
  ill-conditioned cases from a flat start. A Gauss-Seidel or DC-power-flow
  initialisation, or a step-length limiter on Δx, would widen the set of cases
  that converge at all.
- **Contingency analysis.** Once the solver is sparse, N−1 screening is the
  obvious application: re-solve with each branch removed and report the
  violations. That is what load flow is actually used for.
- **Report sensitivities.** The Jacobian at the solution is already the linear
  sensitivity of injections to voltages; inverting the relationship gives
  voltage sensitivity to injection, which is free information the solver
  currently throws away.

## License

MIT.
