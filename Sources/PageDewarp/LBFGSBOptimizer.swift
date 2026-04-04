// LBFGSBOptimizer.swift
// Swift wrapper around the vendored L-BFGS-B-C library.
// Uses finite-difference numerical gradients.
//
// The C library is the same code SciPy wraps internally (BSD-3 licensed).
// See Sources/LBFGSB/LICENSE for the full license text.

import Foundation
#if SWIFT_PACKAGE
import CLBFGSB
#endif

// MARK: - Finite-Difference Gradient

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

// MARK: - L-BFGS-B Minimize

/// Minimizes a scalar function using L-BFGS-B via the vendored C library.
///
/// Uses finite-difference numerical gradients. Parameters are unconstrained.
/// Matches SciPy's `minimize(method='L-BFGS-B')` configuration.
///
/// - Parameters:
///   - objective: Scalar objective function `([Double]) -> Double`.
///   - x0: Initial parameter vector.
///   - maxIter: Maximum number of iterations (default 600,000).
///   - maxCor: Number of L-BFGS correction pairs (default 100).
///   - factr: Function value tolerance factor. Iteration stops when
///     `(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= factr*epsmch`.
///     Default `1e7` for moderate accuracy. Use `1e12` for low, `1e1` for high.
///   - pgtol: Projected gradient tolerance (default 1e-5).
///   - epsilon: Finite difference step size (default 1e-8).
/// - Returns: Optimization result.
func lbfgsbMinimize(
    objective: ([Double]) -> Double,
    x0: [Double],
    maxIter: Int = 600_000,
    maxCor: Int = 100,
    factr: Double = 1e7,
    pgtol: Double = 1e-5,
    epsilon: Double = 1e-8
) -> OptimizeResult {
    let n = x0.count

    // Parameter vector (in/out)
    var x = x0

    // Bounds — all unconstrained (nbd[i] = 0)
    var l = [Double](repeating: 0.0, count: n)
    var u = [Double](repeating: 0.0, count: n)
    var nbd = [Int](repeating: 0, count: n)

    // Function value and gradient (in/out)
    var f: Double = 0.0
    var g = [Double](repeating: 0.0, count: n)

    // Tolerances
    var factrVar = factr
    var pgtolVar = pgtol

    // Working arrays
    // wa size: (2*m + 5)*n + 11*m^2 + 8*m (from driver1.c documentation)
    let m = maxCor
    let waSize = (2 * m + 5) * n + 11 * m * m + 8 * m
    var wa = [Double](repeating: 0.0, count: waSize)
    var iwa = [Int](repeating: 0, count: 3 * n)

    // Task and csave are single integer values (not arrays/strings).
    // The C library uses integer codes defined in lbfgsb.h:
    //   START=1, NEW_X=2, FG=10..15 (IS_FG), CONVERGENCE=20..25, ERROR=200+
    var task: Int = 1  // START
    var csave: Int = 0

    // Internal state arrays
    var lsave = [Int](repeating: 0, count: 4)
    var isave = [Int](repeating: 0, count: 44)
    var dsave = [Double](repeating: 0.0, count: 29)

    var iprint: Int = -1  // silent
    var nVar = n
    var mVar = m
    var nfev = 0
    var converged = false

    // Reverse-communication loop
    for _ in 0..<(maxIter + 1) {
        setulb(&nVar, &mVar, &x, &l, &u, &nbd,
               &f, &g, &factrVar, &pgtolVar,
               &wa, &iwa, &task, &iprint,
               &csave, &lsave, &isave, &dsave)

        if task >= 10 && task <= 15 {
            // IS_FG: evaluate function and gradient
            f = objective(x)
            nfev += 1
            g = finiteDifferenceGradient(
                objective: objective, x: x, f0: f, epsilon: epsilon
            )
            nfev += n
        } else if task == 2 {
            // NEW_X: new iterate accepted, continue
            continue
        } else if task >= 20 && task <= 25 {
            // IS_CONVERGED
            converged = true
            break
        } else if task >= 100 || task == 3 || task == 4 {
            // WARNING, ERROR, ABNORMAL, or RESTART — stop
            break
        } else {
            break
        }
    }

    return OptimizeResult(x: x, fun: f, nfev: nfev, converged: converged)
}

// MARK: - L-BFGS-B with Analytical Gradient

/// Minimizes a scalar function using L-BFGS-B with a caller-provided gradient.
///
/// The `objectiveAndGradient` closure returns both the function value and gradient
/// in a single call, eliminating the need for finite-difference approximation.
///
/// - Parameters:
///   - objectiveAndGradient: Returns `(f, grad)` for a given parameter vector.
///   - x0: Initial parameter vector.
///   - maxIter: Maximum number of iterations (default 600,000).
///   - maxCor: Number of L-BFGS correction pairs (default 100).
///   - factr: Function value tolerance factor (default 1e7).
///   - pgtol: Projected gradient tolerance (default 1e-5).
/// - Returns: Optimization result.
func lbfgsbMinimize(
    objectiveAndGradient: ([Double]) -> (f: Double, grad: [Double]),
    x0: [Double],
    maxIter: Int = 600_000,
    maxCor: Int = 100,
    factr: Double = 1e7,
    pgtol: Double = 1e-5
) -> OptimizeResult {
    let n = x0.count
    var x = x0
    var l = [Double](repeating: 0.0, count: n)
    var u = [Double](repeating: 0.0, count: n)
    var nbd = [Int](repeating: 0, count: n)
    var f: Double = 0.0
    var g = [Double](repeating: 0.0, count: n)
    var factrVar = factr
    var pgtolVar = pgtol
    let m = maxCor
    let waSize = (2 * m + 5) * n + 11 * m * m + 8 * m
    var wa = [Double](repeating: 0.0, count: waSize)
    var iwa = [Int](repeating: 0, count: 3 * n)
    var task: Int = 1  // START
    var csave: Int = 0
    var lsave = [Int](repeating: 0, count: 4)
    var isave = [Int](repeating: 0, count: 44)
    var dsave = [Double](repeating: 0.0, count: 29)
    var iprint: Int = -1
    var nVar = n
    var mVar = m
    var nfev = 0
    var converged = false

    for _ in 0..<(maxIter + 1) {
        setulb(&nVar, &mVar, &x, &l, &u, &nbd,
               &f, &g, &factrVar, &pgtolVar,
               &wa, &iwa, &task, &iprint,
               &csave, &lsave, &isave, &dsave)

        if task >= 10 && task <= 15 {
            // IS_FG: evaluate function and gradient together
            let result = objectiveAndGradient(x)
            f = result.f
            g = result.grad
            nfev += 1
        } else if task == 2 {
            continue
        } else if task >= 20 && task <= 25 {
            converged = true
            break
        } else if task >= 100 || task == 3 || task == 4 {
            break
        } else {
            break
        }
    }

    return OptimizeResult(x: x, fun: f, nfev: nfev, converged: converged)
}
