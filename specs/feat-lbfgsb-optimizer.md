# Feature: Replace Powell with L-BFGS-B Optimizer

- **Status**: Draft
- **Authors**: Claude Code, 2026-03-30
- **Type**: Feature (performance + quality)

---

## Overview

Replace the Powell (gradient-free) optimizer in the Swift iOS page-dewarp implementation with L-BFGS-B using finite-difference numerical gradients. This matches the Python reference's optimization algorithm, improving both speed (7-15x) and output quality (same optimizer = same minima).

## Background / Problem Statement

The Swift implementation currently uses Powell's conjugate direction method (gradient-free), while the Python reference uses L-BFGS-B with JAX autodiff gradients. This causes two problems:

1. **Speed**: Powell needs up to 600K function evaluations (2-50s per image). Python's L-BFGS-B converges in ~100-300 evaluations (0.3-0.5s with JAX).
2. **Quality**: Powell and L-BFGS-B find different local minima, causing an 81-93% pixel match gap between Swift and Python output. Using the same optimizer should close this gap significantly.

JAX is not available on iOS, but L-BFGS-B does not require autodiff -- it just needs gradients. Finite-difference numerical gradients (evaluating `f(x + eps)` for each parameter) are a standard approach when analytical gradients are unavailable.

**Cost analysis**: With ~400 parameters, each gradient computation needs ~800 function evaluations (forward differences) or ~400 (if we use the current `f(x)` value). L-BFGS-B typically converges in 50-100 outer iterations, giving 20K-80K total evaluations. This is still fewer than Powell's 600K maximum and each evaluation is the same cheap projection + sum-of-squares.

## Goals

- Replace Powell with L-BFGS-B for the main parameter optimization in `DewarpPipeline.swift`
- Use finite-difference numerical gradients (no autodiff dependency)
- Match Python's L-BFGS-B configuration: `maxiter=600_000`, `maxcor=100`
- Achieve faster convergence than Powell (target: <5s per image on simulator)
- Improve pixel match with Python reference (target: >93% average)
- Keep Powell available for the 2D page-dimension optimization (only 2 parameters; L-BFGS-B is overkill)

## Non-Goals

- Implementing automatic differentiation in Swift
- Analytical (hand-coded) Jacobian computation (future optimization)
- GPU acceleration of the objective function
- Changing the objective function itself

## Technical Dependencies

- No new external dependencies. L-BFGS-B is implemented from scratch in Swift.
- Existing `Objective.swift` and `Projection.swift` are unchanged.
- Existing `PowellOptimizer.swift` is retained (used for page-dim optimization).

## Detailed Design

### New File: `LBFGSBOptimizer.swift`

A self-contained L-BFGS-B implementation in `Sources/Core/`. The algorithm has three components:

#### 1. Finite-Difference Gradient

```swift
/// Compute gradient via forward finite differences.
/// Cost: N+1 function evaluations (reuses f(x) passed in as f0).
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

Uses forward differences with adaptive step size `h = max(eps, eps * |x_i|)`. Reuses `f(x)` to save one evaluation per gradient.

#### 2. L-BFGS Two-Loop Recursion

The core L-BFGS algorithm maintains a history of the last `m` correction pairs `(s_k, y_k)` where:
- `s_k = x_{k+1} - x_k` (parameter step)
- `y_k = g_{k+1} - g_k` (gradient change)

The two-loop recursion computes the search direction `d = -H_k * g_k` without explicitly forming the inverse Hessian:

```swift
func lbfgsTwoLoop(
    gradient: [Double],
    sHistory: [[Double]],   // past m steps
    yHistory: [[Double]],   // past m gradient changes
    rhoHistory: [Double]    // 1 / dot(y_k, s_k)
) -> [Double] {
    let m = sHistory.count
    var q = gradient
    var alphas = [Double](repeating: 0.0, count: m)

    // First loop: backward
    for i in stride(from: m - 1, through: 0, by: -1) {
        alphas[i] = rhoHistory[i] * dot(sHistory[i], q)
        q = q - alphas[i] * yHistory[i]  // vectorized
    }

    // Scale by initial Hessian approximation: H0 = gamma * I
    // gamma = dot(s_{m-1}, y_{m-1}) / dot(y_{m-1}, y_{m-1})
    var r: [Double]
    if m > 0 {
        let gamma = dot(sHistory[m-1], yHistory[m-1]) / dot(yHistory[m-1], yHistory[m-1])
        r = q.map { $0 * gamma }
    } else {
        r = q
    }

    // Second loop: forward
    for i in 0..<m {
        let beta = rhoHistory[i] * dot(yHistory[i], r)
        r = r + (alphas[i] - beta) * sHistory[i]  // vectorized
    }

    return r.map { -$0 }  // negate for descent direction
}
```

#### 3. Box-Constrained Line Search (Wolfe Conditions)

L-BFGS-B uses a backtracking line search satisfying the strong Wolfe conditions:

```swift
func wolfeLineSearch(
    objective: ([Double]) -> Double,
    gradient: ([Double], Double) -> [Double],
    x: [Double],
    direction: [Double],
    f0: Double,
    g0: [Double],
    c1: Double = 1e-4,    // Armijo condition
    c2: Double = 0.9,     // Curvature condition
    maxIter: Int = 20
) -> (x: [Double], f: Double, g: [Double], nfev: Int)
```

The "B" (box-constrained) part is not needed for our problem since parameters are unconstrained (the cubic coefficients are clamped inside the projection function, not by the optimizer).

#### 4. Main Entry Point

```swift
func lbfgsbMinimize(
    objective: ([Double]) -> Double,
    x0: [Double],
    maxIter: Int = 600_000,
    maxCor: Int = 100,
    ftol: Double = 2.220446049250313e-16,  // machine epsilon (SciPy default)
    gtol: Double = 1e-5,                    // gradient norm tolerance (SciPy default)
    epsilon: Double = 1e-8                   // finite difference step
) -> OptimizeResult
```

**Algorithm outline:**
```
x = x0
f = objective(x)
g = finiteDifferenceGradient(objective, x, f, epsilon)
sHistory, yHistory, rhoHistory = []  // bounded deques of size maxCor

for k in 0..<maxIter:
    // Check gradient convergence
    if max(|g_i|) <= gtol:
        break

    // Compute search direction via two-loop recursion
    d = lbfgsTwoLoop(g, sHistory, yHistory, rhoHistory)

    // Line search
    (x_new, f_new, g_new, nfev) = wolfeLineSearch(objective, gradient, x, d, f, g)

    // Check function value convergence
    if |f - f_new| / max(|f|, |f_new|, 1) <= ftol:
        break

    // Update history
    s = x_new - x
    y = g_new - g
    rho = 1.0 / dot(y, s)
    if rho > 0:  // skip update if curvature condition violated
        append(sHistory, s); append(yHistory, y); append(rhoHistory, rho)
        if len > maxCor: drop oldest

    x = x_new; f = f_new; g = g_new

return OptimizeResult(x: x, fun: f, nfev: totalEvals, converged: true)
```

### Changes to `DewarpPipeline.swift`

Replace the Powell call for the main optimization:

```swift
// Before:
let optResult = powellMinimize(objective: objective, x0: initialParams)

// After:
let optResult = lbfgsbMinimize(objective: objective, x0: initialParams)
```

Keep Powell for the 2D page-dimension optimization (line 217):
```swift
// Unchanged -- only 2 parameters, Powell is fine
let result = powellMinimize(objective: dimObjective, x0: dims)
```

### File Organization

```
Sources/Core/
  LBFGSBOptimizer.swift    (NEW)
  PowellOptimizer.swift     (UNCHANGED, still used for page-dim optimization)
  DewarpPipeline.swift      (ONE LINE CHANGE: powellMinimize → lbfgsbMinimize)
  Objective.swift            (UNCHANGED)
  Projection.swift           (UNCHANGED)
```

### Reuse of Existing Types

The new optimizer returns the same `OptimizeResult` struct already defined in `PowellOptimizer.swift`:

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

No user-facing API changes. Images are dewarped faster and with slightly better quality. The `DewarpPipeline.process(image:)` call is unchanged.

## Testing Strategy

### Unit Tests (`LBFGSBOptimizerTests.swift`)

**1. Gradient computation accuracy**
```swift
// Test: finiteDifferenceGradient on f(x) = x[0]^2 + x[1]^2
// Expected gradient at (3, 4): [6, 8]
// Tolerance: 1e-4 (limited by finite difference precision)
```

**2. Rosenbrock function** (standard optimizer benchmark)
```swift
// f(x,y) = (1-x)^2 + 100*(y-x^2)^2
// Start at (-1, 1), expect convergence to (1, 1) within 1e-4
```

**3. High-dimensional quadratic** (mimics real workload)
```swift
// f(x) = sum((x[i] - i)^2) for i in 0..<100
// Verify convergence in fewer evaluations than Powell
```

**4. Comparison with Powell on same objective**
```swift
// Both optimizers start from same x0 on the real dewarp objective
// L-BFGS-B should achieve equal or lower final loss
// L-BFGS-B should use fewer function evaluations
```

### Integration Tests

**5. Pipeline end-to-end with golden file**
```swift
// Run DewarpPipeline.process on boston_cooking_a_input.jpg
// Verify output is non-nil and dimensions are reasonable
// Compare against golden output (should be closer to Python than Powell was)
```

**6. Comparison test on 5 test images**
```swift
// Run on IMG_1358-1362, save outputs
// Compare pixel match with Python reference
// Verify average pixel match > 93% (up from ~89% with Powell)
```

### Regression Tests

All existing tests in `PowellOptimizerTests.swift` remain unchanged and must pass (Powell is still used for page-dim optimization).

## Performance Considerations

**Expected improvement:**
- Powell: ~600K max evals, typically 50K-200K for convergence
- L-BFGS-B with numerical gradients: ~50-100 outer iterations x (N+1) evals per gradient ≈ 20K-40K total evals
- Each evaluation is the same cost (projection + sum of squares)
- Net speedup: 3-10x

**Memory:**
- L-BFGS-B stores `maxCor=100` correction pairs, each of size N (~400)
- Total: 100 * 400 * 2 * 8 bytes ≈ 640KB. Negligible.

**Gradient computation overhead:**
- Each gradient needs N evaluations of the objective
- With N=400, this is 400 evaluations per outer step
- But L-BFGS-B needs ~50-100 outer steps total, not per parameter

## Security Considerations

None. Pure numerical computation, no I/O, no user data beyond the input image.

## Documentation

- Update `docs/architecture.md` to note L-BFGS-B for main optimization
- Update `docs/differences.md` to note Swift now uses the same optimizer algorithm as Python (but with numerical rather than analytical gradients)

## Implementation Phases

### Phase 1: Core L-BFGS-B Implementation
- Implement `finiteDifferenceGradient`
- Implement `lbfgsTwoLoop`
- Implement `wolfeLineSearch`
- Implement `lbfgsbMinimize`
- Unit tests for each component
- Unit tests for standard optimization benchmarks (Rosenbrock, quadratic)

### Phase 2: Integration
- Switch `DewarpPipeline.swift` from `powellMinimize` to `lbfgsbMinimize`
- Run pipeline integration test on golden file
- Run comparison test on 5 images, measure pixel match improvement

### Phase 3: Validation
- Run on full 20-image test pool
- Compare performance (time per image) against Powell
- Compare pixel match against Python reference
- Update documentation

## Open Questions

1. **Forward vs central differences?** Forward differences need N extra evals per gradient. Central differences need 2N but are more accurate. Start with forward (cheaper), switch to central if gradient quality is insufficient.

2. **Should we use Accelerate/vDSP for vector operations?** The two-loop recursion involves many `dot` and `axpy` operations on vectors of size ~400. Accelerate could speed these up, but the bottleneck is the N=400 objective evaluations per gradient, not the linear algebra. Defer to a future optimization pass.

3. **Convergence tolerance values?** SciPy's L-BFGS-B defaults are `ftol=2.22e-16` (machine epsilon) and `gtol=1e-5`. The Python page-dewarp code only overrides `maxiter` and `maxcor`, accepting these defaults. We should match them.

## References

- SciPy L-BFGS-B source: `scipy/optimize/_lbfgsb_py.py`
- Nocedal & Wright, "Numerical Optimization", Chapter 7 (L-BFGS)
- Python reference: `src/page_dewarp/optimise/_jax.py` (L-BFGS-B config)
- Python config: `MAX_CORR=100`, `OPT_MAX_ITER=600_000`
- Current Swift optimizer: `ios/PageDewarp/Sources/Core/PowellOptimizer.swift`
- Current Swift objective: `ios/PageDewarp/Sources/Core/Objective.swift`
