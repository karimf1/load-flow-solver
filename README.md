# Newton-Raphson load flow solver

A power flow solver written from scratch in C++17: assemble the bus admittance
matrix, derive the power-flow Jacobian by hand, and solve for the steady-state
voltage at every bus with Newton-Raphson. Validated against the IEEE 14-bus test
system.

**Converges IEEE 14-bus in 4 iterations, matching the published solution to
5e-4 p.u. — the resolution the reference is quoted to.**

## Validation

| quantity | computed | published |
|---|---|---|
| max \|ΔV\| across all buses | 4.9e-4 p.u. | (reference quoted to 3 dp) |
| max \|Δangle\| across all buses | 4.8e-4 deg | — |
| slack generation | 232.393 MW | 232.4 MW |
| total real losses | 13.393 MW | 13.393 MW |
| branch 1–2 flow | 156.88 MW | 156.88 MW |
| branch 1–5 flow | 75.51 MW | 75.51 MW |

## Build and run

Two files, no build system. You need a C++17 compiler and Eigen's headers —
`brew install eigen` or `apt install libeigen3-dev`.

```bash
g++ -std=c++17 -O2 -I/opt/homebrew/include/eigen3 nrlf.cpp -o nrlf
```

On Linux that include path is usually `/usr/include/eigen3`.

```bash
./nrlf case14              # solve, print the voltage profile
./nrlf case14 --flows      # add the per-branch flow table
./nrlf case14 -v --ybus    # mismatch each iteration, and the Y-bus
./nrlf case14 --plot .     # write voltage_profile.svg and convergence.svg
./nrlf --test              # run all 193 checks
```

On a log axis the convergence plot bends steadily downward rather than running
straight: each Newton step roughly squares the previous error.

## How it works

**Y-bus.** Each branch stamps a 2×2 admittance into a complex matrix: `−y/a`
off-diagonal and `y + jb/2` on the diagonal, with the tap-side diagonal scaled
by `1/a²` for transformers. Bus shunts add `Gs + jBs`. Stamps accumulate, so
parallel branches add rather than overwrite.

**Newton-Raphson.** Power is quadratic in voltage, so
`P_i = V_i Σ_k V_k (G cos θ_ik + B sin θ_ik)` has no closed form. The solver
linearizes with the Jacobian

```
J = [ ∂P/∂θ   ∂P/∂|V| ]
    [ ∂Q/∂θ   ∂Q/∂|V| ]
```

solves `J Δx = mismatch` with Eigen's LU, and iterates from a flat start.
Unknowns are θ at every non-slack bus and |V| at every PQ bus — 22 unknowns for
the 14-bus system. The diagonal Jacobian terms collapse into the already-computed
injections (`∂P_i/∂θ_i = −Q_i − B_ii V_i²` and friends), which is both cheaper
and far easier to get right than differentiating the sum directly.

**Branch flows** reuse the same `branch_admittance()` the matrix was built from,
so the flow report cannot drift out of sync with the model the solver used.

## Testing

193 checks, run with `./nrlf --test`. Two are load-bearing:

**Manufactured solution.** Pick an arbitrary voltage state, compute the power
injections it implies, feed those back as the scheduled loads, and require the
solver to recover the original voltages to 1e-9. This validates the Jacobian
against the injection equations using no external data at all — a wrong partial
derivative either misses the target or destroys the quadratic convergence.

**Per-bus conservation of complex power.** At every bus, the power flowing away
into the incident branches plus what the local shunt absorbs must equal the
power injected there, to 1e-9. This ties the branch flow model back to the
Y-bus the solver actually used; a sign error or dropped tap term in either one
would have to be wrong in exactly compensating ways to survive at all 14 buses.

Also checked: published reference values, power balance, PV/slack bus
constraints, an unloaded network solving exactly flat, per-branch loss verified
independently as I²R, quadratic convergence, and named errors for malformed
cases.

## Scope

Balanced positive-sequence steady state. Deliberately not included: optimal
power flow, dynamics and stability, three-phase unbalanced modelling, and
generator reactive limits. Dense matrices throughout — real grids are ~99%
sparse and production solvers exploit that, but at 14 buses it buys nothing.

## Layout

Two files by design: this README and `nrlf.cpp`. The single translation unit
holds the types, Y-bus assembly, the Newton-Raphson solver, branch flows, the
SVG writer, both test networks, and the whole test suite, in that reading
order. It compiles in one command with no build system, and `--test` runs what
used to be three separate test binaries.

The cost of that choice is one 1,566-line file instead of a `src/` tree, and no
CI workflow — `./nrlf --test` is the check, run by hand.

## License

MIT.

```
MIT License

Copyright (c) 2026 Karim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
