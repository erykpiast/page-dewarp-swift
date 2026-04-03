# Feature: Replace Powell with L-BFGS-B Optimizer

- **Status**: Draft
- **Authors**: Claude Code, 2026-04-03 (revision of 2026-03-30 draft)
- **Type**: Feature (performance + accuracy)

---

## Overview

Replace the Powell (gradient-free) optimizer with L-BFGS-B for the main parameter optimization in the Swift page-dewarp pipeline. This is the single highest-impact change for closing the accuracy gap with the Python reference and dramatically improving processing speed.

## Background / Problem Statement

The Swift port uses Powell's conjugate direction method (`PowellOptimizer.swift`), while the Python reference uses `scipy.optimize.minimize(method='L-BFGS-B')` with JAX autodiff gradients. This mismatch is the **primary source** of two problems:

1. **Accuracy gap**: Powell and L-BFGS-B converge to different local minima. Current pixel match with Python: 81–96% (avg ~89%). This undermines confidence in the port's correctness because it's impossible to distinguish optimizer-caused differences from actual bugs.

2. **Speed gap**: Powell needs up to 600K function evaluations (2–17s per image on device). Python's L-BFGS-B converges in ~100–300 gradient evaluations (0.3–0.5s with JAX). Even with numerical gradients (no JAX on iOS), L-BFGS-B should be 3–10x faster.

### Why the optimizer matters for accuracy

The objective function (`Objective.swift:26–43`) is a sum of squared projection errors — a smooth, differentiable landscape with many local minima. Powell explores this landscape via 1D line searches along conjugate directions, while L-BFGS-B uses gradient information to navigate more directly. The two methods systematically find different optima. By using the same algorithm as Python, we eliminate this variable and can isolate any remaining differences to actual implementation bugs.

### Parameter vector structure

From `Solver.swift:72–79`, the parameter vector is:
```
pvec = [rvec(3), tvec(3), cubic(2), ycoords(nspans), xcoords(npts)]
```
Total size: `8 + nspans + npts`, typically **89–600 parameters** (~400 typical). Each objective evaluation calls through the ObjC++ bridge to OpenCV's `projectPoints`.

## Goals

- Replace Powell with L-BFGS-B for the main optimization (`DewarpPipeline.swift:109`)
- Use finite-difference numerical gradients (JAX unavailable on iOS)
- Match Python's L-BFGS-B configuration: `maxiter=600_000`, `maxcor=100`
- Improve pixel match with Python reference (target: >93% average, up from ~89%)
- Achieve faster convergence than Powell (target: <5s per image)
- Keep Powell for the 2D page-dimension optimization (`DewarpPipeline.swift:217`, only 2 params)

## Non-Goals

- Implementing automatic differentiation (JAX/Swift AD)
- Analytical (hand-coded) Jacobian computation
- GPU acceleration of the objective function
- Changing the objective function or projection model
- Box constraints (the "B" in L-BFGS-B — our parameters are unconstrained; cubic coefficients are clamped inside `Projection.swift:19–20`, not by the optimizer)

## Technical Dependencies — Implementation Options Analysis

### Option A: Pure Swift implementation (FALLBACK)

Write L-BFGS-B from scratch in Swift. The core algorithm is ~200 lines: two-loop recursion + Wolfe line search + history management.

| Aspect | Assessment |
|--------|------------|
| **License** | N/A — our own code |
| **Integration** | Native Swift, no bridging |
| **Maintenance** | Full control, easy to debug |
| **Correctness** | Can be validated against SciPy's output on known inputs |
| **Complexity** | Medium — the algorithm is well-documented (Nocedal & Wright Ch. 7) |
| **Risk** | Subtle numerical bugs possible; mitigated by extensive testing |

**Why this is recommended**: The algorithm is compact and well-understood. A pure Swift implementation avoids build complexity, is debuggable, and doesn't introduce new dependencies into an iOS app. The two-loop recursion is ~30 lines; the line search is ~50 lines; finite-difference gradient is ~15 lines.

### Option B: Wrap `L-BFGS-B-C` (stephenbeckr/L-BFGS-B-C)

This is the same Fortran→C port (via f2c) that SciPy wraps internally. BSD-3 licensed.

| Aspect | Assessment |
|--------|------------|
| **License** | BSD-3 — App Store compatible |
| **Integration** | Drop `lbfgsb.c` + `miniCBLAS.c` into Xcode, create bridging header |
| **Maintenance** | Mature, stable (last significant change: Fortran v3.0 from 2011) |
| **Correctness** | Exact same code as SciPy — highest fidelity to Python reference |
| **Complexity** | Low — thin C bridging layer; the f2c internals are a black box we don't need to read |
| **Risk** | Low — battle-tested code used by SciPy in production for decades |

**Strong candidate**: Maximum fidelity to SciPy's behavior (literally the same C code underneath). We treat it as a black box — all we write is a thin Swift→C bridge to call `setulb()`. No need to understand the f2c internals.

### Option C: `LBFGSpp` C++ header-only (yixuan/LBFGSpp)

MIT-licensed C++ library with `LBFGSBSolver` class. Depends on Eigen.

| Aspect | Assessment |
|--------|------------|
| **License** | MIT — App Store compatible |
| **Integration** | Xcode 15+ Swift/C++ interop, or ObjC++ wrapper |
| **Maintenance** | Active, well-maintained |
| **Correctness** | Independent implementation, well-tested |
| **Complexity** | High — Eigen dependency is heavy (~15MB headers), C++ interop adds friction |
| **Risk** | Eigen compile times; C++ interop is still evolving in Swift |

**Not recommended**: The Eigen dependency is disproportionate for calling a single optimization function.

### Option D: `liblbfgs` (chokkan/liblbfgs)

Pure C, MIT-licensed L-BFGS implementation. **No box constraints** (L-BFGS only, not L-BFGS-B).

| Aspect | Assessment |
|--------|------------|
| **License** | MIT |
| **Integration** | Simple C bridging |
| **Correctness** | L-BFGS only (no bounds), but we don't need bounds |
| **Complexity** | Low |
| **Risk** | No bound constraints means different convergence path than SciPy's L-BFGS-B |

**Viable alternative** if we confirm bounds are truly unnecessary. Since `Projection.swift:19–20` clamps cubic coefficients internally, the optimizer doesn't need bounds. However, SciPy's L-BFGS-B has additional safeguards (e.g., Cauchy point computation) that plain L-BFGS lacks.

### Option E: Apple Accelerate framework

Accelerate provides BLAS/LAPACK/vDSP but **no optimization routines**. `BNNS.AdamOptimizer` exists but is for neural network training only. Not viable.

### Recommendation

**Option B (wrap L-BFGS-B-C)** for these reasons:
1. **Maximum accuracy**: Identical C code to what SciPy uses — eliminates optimizer implementation as a variable when comparing Swift vs Python output
2. **Battle-tested**: This code has been in production via SciPy for decades; no risk of subtle numerical bugs in our own implementation
3. **Minimal integration effort**: Two C files (`lbfgsb.c`, `miniCBLAS.c`) + a thin Swift bridging layer
4. **The project already has an ObjC++ bridge**: `OpenCVWrapper.mm` demonstrates the pattern; adding a C bridge is simpler than ObjC++
5. **Easy to instrument**: We control the calling layer and can log iteration count, gradient norms, etc.

**Option A (pure Swift) is the fallback** if Option B proves harder to integrate than expected (e.g., Fortran-style array conventions causing issues at the bridging layer).

## Detailed Design

### Approach: Wrap L-BFGS-B-C via C bridging

The design has two layers:
1. **Vendored C code** — `lbfgsb.c` and `miniCBLAS.c` from `stephenbeckr/L-BFGS-B-C`, added to the Xcode project as C sources
2. **Swift wrapper** — `LBFGSBOptimizer.swift` that provides a clean Swift API, handles finite-difference gradient computation, and calls the C `setulb()` function

The C library's entry point is `setulb()`, which uses a reverse-communication interface: it returns control to the caller requesting either a function evaluation or a gradient evaluation, then is called again with the results. This maps cleanly to our finite-difference gradient approach.

### New Files

```
Sources/
  LBFGSB/                      (NEW directory)
    lbfgsb.c                   (vendored from L-BFGS-B-C, BSD-3)
    lbfgsb.h                   (vendored)
    miniCBLAS.c                (vendored)
    miniCBLAS.h                (vendored)
    LICENSE                    (BSD-3 license text)
  Core/
    LBFGSBOptimizer.swift      (NEW — Swift wrapper + finite-difference gradient)
```

### Finite-Difference Gradient (in Swift wrapper)

```swift
/// Compute gradient via forward finite differences.
/// Cost: N function evaluations (reuses f(x) passed in as f0).
func finiteDifferenceGradient(
    objective: ([Double]) -> Double,
    x: [Double],
    f0: Double,
    epsilon: Double = 1e-8
) -> [Double] {
    let n = x.count
    var grad = [Double](repeating: 0.0, count: n)
    var xp = x
    for i in 0..<n {
        let xi = x[i]
        let h = max(epsilon, epsilon * abs(xi))
        xp[i] = xi + h
        let fi = objective(xp)
        grad[i] = (fi - f0) / h
        xp[i] = xi
    }
    return grad
}
```

**Important note on ObjC++ bridge overhead**: Each `objective(xp)` call goes through `projectKeypoints` → `projectXY` → `OpenCVWrapper.projectPointsWith3DPoints` (ObjC++ bridge). With N=400 parameters, each gradient computation makes 400 bridge crossings. This is the computational bottleneck. The bridge overhead per call is small (~0.01ms), so 400 calls ≈ 4ms per gradient, and 50–100 gradients ≈ 200–400ms total bridge overhead.

### Swift Wrapper: `LBFGSBOptimizer.swift`

The wrapper handles three responsibilities:
1. Finite-difference gradient computation (in Swift)
2. Translating between Swift types and C arrays
3. Driving the `setulb()` reverse-communication loop

#### `setulb()` Reverse-Communication Interface

The C library uses a reverse-communication pattern — the caller allocates working arrays and calls `setulb()` in a loop. On each return, `setulb()` sets a `task` string indicating what it needs:

```
task = "START"
while true:
    setulb(&n, &m, x, l, u, nbd, &f, g, ...)  // C call
    if task starts with "FG":
        f = objective(x)           // evaluate function
        g = gradient(x, f)         // compute gradient (finite differences)
    elif task starts with "NEW_X":
        continue                   // new iterate accepted, continue
    elif task starts with "CONV":
        break                      // converged
    elif task starts with "ABNO" or "ERROR":
        break                      // abnormal termination
```

This maps cleanly to our needs: when `setulb` asks for "FG", we compute `f` via our existing `objective` closure and `g` via `finiteDifferenceGradient`.

#### `setulb()` C Signature and Working Arrays

```c
void setulb_(int *n, int *m, double *x, double *l, double *u, int *nbd,
             double *f, double *g, double *factr, double *pgtol,
             double *wa, int *iwa, char *task, int *iprint,
             char *csave, int *lsave, int *isave, double *dsave);
```

Key parameters:
- `n`: number of variables (our parameter count, ~400)
- `m`: number of correction pairs (`maxCor = 100`)
- `x[n]`: parameter vector (in/out)
- `l[n]`, `u[n]`: lower/upper bounds (unused — set `nbd[i] = 0` for all i)
- `f`, `g[n]`: function value and gradient (in/out, filled by caller on "FG" tasks)
- `factr`: function tolerance factor (`ftol / machine_epsilon`)
- `pgtol`: projected gradient tolerance (`gtol`)
- `wa[(2*m + 5)*n + 12*m*(m+1)]`: working array — for m=100, n=400: **~1.3MB**
- `iwa[3*n]`: integer working array — for n=400: **~5KB**
- `task[60]`: reverse-communication task string (in/out)
- `iprint`: verbosity (-1 = silent)
- `csave[60]`, `lsave[4]`, `isave[44]`, `dsave[29]`: internal state arrays

All arrays are caller-allocated. The Swift wrapper allocates them once before the loop and deallocates after convergence.

#### Swift API

```swift
func lbfgsbMinimize(
    objective: ([Double]) -> Double,
    x0: [Double],
    maxIter: Int = 600_000,
    maxCor: Int = 100,
    ftol: Double = 2.220446049250313e-16,  // machine epsilon (matches SciPy)
    gtol: Double = 1e-5,                    // gradient norm tolerance (matches SciPy)
    epsilon: Double = 1e-8                   // finite difference step
) -> OptimizeResult
```

#### Bound constraints

All parameters are unconstrained (`nbd[i] = 0` for all i). The cubic coefficients are clamped inside `Projection.swift:19–20`, not by the optimizer. This matches Python's usage (no bounds passed to `scipy.optimize.minimize`).

### Changes to `DewarpPipeline.swift`

Single-line change at line 109:

```swift
// Before (line 109):
let optResult = powellMinimize(objective: objective, x0: initialParams)

// After:
let optResult = lbfgsbMinimize(objective: objective, x0: initialParams)
```

Powell remains for page-dimension optimization (line 217, only 2 parameters).

### Complete File Organization

```
Sources/
  LBFGSB/                      (NEW — vendored C library)
    lbfgsb.c                   (from stephenbeckr/L-BFGS-B-C)
    lbfgsb.h                   (bridging header for setulb)
    miniCBLAS.c                (minimal BLAS routines used by lbfgsb)
    miniCBLAS.h
    LICENSE                    (BSD-3 license)
  Core/
    LBFGSBOptimizer.swift      (NEW — ~100 lines: Swift wrapper + gradient)
    PowellOptimizer.swift       (UNCHANGED)
    DewarpPipeline.swift        (ONE LINE CHANGE)
    Objective.swift              (UNCHANGED)
    Projection.swift             (UNCHANGED)
```

### Reuse of Existing Types

Returns the same `OptimizeResult` struct from `PowellOptimizer.swift:11–16`:

```swift
struct OptimizeResult {
    let x: [Double]
    let fun: Double
    let nfev: Int
    let converged: Bool
}
```

No API changes. `DewarpPipeline.process(image:)` signature is unchanged.

## User Experience

No user-facing API changes. Same `DewarpPipeline.process(image:)` call. Users observe:
- Faster processing (target: 3–10x improvement)
- Slightly different (closer to Python reference) output pixels
- No behavioral changes

## Testing Strategy

### Unit Tests (`LBFGSBOptimizerTests.swift`)

**1. Gradient accuracy** — validates `finiteDifferenceGradient` against known analytical gradient
```swift
// f(x) = x[0]^2 + x[1]^2, grad at (3,4) = [6,8]
// Purpose: catch off-by-one or step-size bugs in gradient computation
// Tolerance: 1e-4 (finite difference precision limit)
```

**2. `setulb()` wrapper round-trip** — validates the Swift↔C bridge
```swift
// Minimize f(x) = x[0]^2 + x[1]^2 from (5, 3) through the full lbfgsbMinimize API
// Must return x ≈ (0, 0), converged = true, nfev > 0
// Purpose: catch bridging bugs (array layout, task-string parsing, working array sizing)
```

**3. Rosenbrock function** — standard nonlinear optimization benchmark
```swift
// f(x,y) = (1-x)^2 + 100*(y-x^2)^2, start at (-1, 1)
// Must converge to (1, 1) within 1e-4
// Purpose: validates the full optimizer on a non-trivial landscape
// This test CAN fail if the line search or two-loop has bugs
```

**4. High-dimensional quadratic** — mimics real parameter count
```swift
// f(x) = sum((x[i] - i)^2) for i in 0..<200
// Must converge in fewer evaluations than Powell on the same problem
// Purpose: validates scaling behavior with realistic N
```

**5. Convergence criteria** — validates stopping conditions
```swift
// Already-optimal x0 → should return immediately (nit=0)
// Very tight gtol → should iterate more than loose gtol
// Purpose: catch bugs where optimizer doesn't stop or stops too early
```

### Integration Tests

**6. Pipeline golden file comparison**
```swift
// Run full pipeline on test image, compare output dimensions and pixel hash
// Purpose: end-to-end validation that the optimizer swap doesn't break the pipeline
```

**7. Python reference comparison on test images**
```swift
// Run on 5+ images, compare pixel match with Python reference output
// Target: average pixel match > 93% (up from ~89% with Powell)
// Purpose: validates that L-BFGS-B actually closes the accuracy gap
// This is the KEY acceptance test for this feature
```

### Regression Tests

All existing `PowellOptimizerTests.swift` must continue to pass (Powell still used for page-dim optimization).

## Performance Considerations

### Expected evaluation counts

| Metric | Powell (current) | L-BFGS-B (projected) |
|--------|-----------------|---------------------|
| Outer iterations | 50K–600K | 50–100 |
| Evals per iteration | 1 (line search) | N+1 ≈ 401 (gradient) |
| Total evaluations | 50K–600K | 20K–40K |
| Time per image | 2–17s | 0.5–5s |

### Memory

L-BFGS-B stores `maxCor=100` correction pairs of size N (~400):
- `100 × 400 × 2 × 8 bytes = 640KB` — negligible

### Bridge overhead

Each objective evaluation crosses the Swift→ObjC++→OpenCV bridge once. At ~0.01ms per crossing:
- Per gradient: `400 × 0.01ms = 4ms`
- Total: `100 gradients × 4ms = 400ms` bridge overhead
- This is small compared to the computation inside each evaluation

### Accelerate/vDSP optimization (deferred)

The two-loop recursion's `dot` and `saxpy` operations on ~400-element vectors could use `cblas_ddot` and `cblas_daxpy`. However, the bottleneck is the N objective evaluations per gradient, not the ~30 vector operations per iteration. Defer to a future pass if profiling shows otherwise.

## Security Considerations

None. Pure numerical computation with no I/O beyond the input image already in memory.

## Documentation

- Update `docs/differences.md` section 3 to note Swift now uses L-BFGS-B (with numerical gradients instead of JAX autodiff)
- Update `docs/architecture.md` if it references the optimizer

## Implementation Phases

### Phase 1: Vendor and Integrate L-BFGS-B-C
- Vendor `lbfgsb.c`, `miniCBLAS.c`, and headers from `stephenbeckr/L-BFGS-B-C`
- Add BSD-3 LICENSE file
- Add C files to Xcode project, configure bridging header
- Implement `LBFGSBOptimizer.swift` — Swift wrapper with `setulb()` reverse-communication loop
- Implement `finiteDifferenceGradient` in Swift
- Unit test: Rosenbrock function convergence
- Unit test: high-dimensional quadratic convergence
- Unit test: convergence criteria (already-optimal x0, tolerance behavior)

### Phase 2: Integration and Validation
- Switch `DewarpPipeline.swift` from `powellMinimize` to `lbfgsbMinimize`
- Run pipeline golden file test
- Run comparison on 5+ test images against Python reference
- Measure pixel match improvement (target: >93% avg)
- Measure processing time improvement

### Phase 3: Tuning and Hardening
- Experiment with forward vs central differences if gradient quality is insufficient
- Add diagnostic logging (iteration count, gradient norm, step size) behind a flag
- Run on full 20-image test pool
- Update documentation

## Open Questions

1. **Forward vs central differences?** Forward differences cost N evals per gradient. Central differences cost 2N but are ~100x more accurate. Start with forward (cheaper); switch to central if gradient quality causes poor convergence. Note: SciPy's own `approx_fprime` uses forward differences by default.

2. **Fallback to Powell**: If L-BFGS-B fails to converge, should we fall back to Powell? Recommend: no fallback in Phase 1; investigate root cause rather than masking.

3. **`miniCBLAS` vs system BLAS**: `L-BFGS-B-C` ships `miniCBLAS.c` for portability. We could instead link against Apple's Accelerate BLAS (`cblas_ddot`, `cblas_daxpy`). Recommend: start with `miniCBLAS` for simplicity; swap to Accelerate only if profiling shows the BLAS calls are a bottleneck (unlikely — the bottleneck is the N objective evaluations per gradient).

4. **Xcode build integration** (RESOLVED): The project uses XcodeGen (`project.yml`) with `sources: - path: Sources`, which auto-includes all files under `Sources/`. The C files in `Sources/LBFGSB/` will be picked up automatically. The existing `module.modulemap` at `Sources/OpenCVBridge/module.modulemap` exposes `OpenCVWrapper.h` to Swift. For the C library, we have two options: (a) add `lbfgsb.h` to the same modulemap, or (b) create a separate `Sources/LBFGSB/module.modulemap`. Recommend (b) — a separate module keeps concerns isolated. The Swift wrapper imports it via `import LBFGSB`.

## References

- **Algorithm**: Nocedal & Wright, *Numerical Optimization*, Chapter 7 (L-BFGS)
- **SciPy source**: `scipy/optimize/_lbfgsb_py.py` — Python wrapper around the Fortran L-BFGS-B v3.0
- **L-BFGS-B original**: Zhu, Byrd, Lu, Nocedal (1997). "Algorithm 778: L-BFGS-B"
- **L-BFGS-B-C**: https://github.com/stephenbeckr/L-BFGS-B-C — BSD-3 C port (fallback Option B)
- **LBFGSpp**: https://github.com/yixuan/LBFGSpp — MIT C++ header-only (Option C)
- **liblbfgs**: https://github.com/chokkan/liblbfgs — MIT C library (Option D)
- **Python reference config**: `MAX_CORR=100`, `OPT_MAX_ITER=600_000` in `src/page_dewarp/options/core.py`
- **Current Swift optimizer**: `Sources/Core/PowellOptimizer.swift`
- **Current Swift objective**: `Sources/Core/Objective.swift`
- **Current Swift projection**: `Sources/Core/Projection.swift`
