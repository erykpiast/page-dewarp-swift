// PowellOptimizer.swift
// Powell's conjugate direction method with Brent's 1D line search.
// Matches scipy.optimize.minimize(method='Powell') behavior.
//
// Ported from scipy/_optimize.py _minimize_powell and brent

import Foundation

// MARK: - Result type

struct OptimizeResult {
    let x: [Double]
    let fun: Double
    let nfev: Int
    let converged: Bool
}

// MARK: - Public API

/// Minimizes a scalar function using Powell's conjugate direction method.
///
/// Matches SciPy `minimize(method='Powell')` behavior.
/// Ported from scipy/_optimize.py:_minimize_powell
///
/// - Parameters:
///   - objective: Scalar objective function.
///   - x0: Initial parameter vector.
///   - maxIter: Maximum number of outer iterations (default 600,000).
///   - ftol: Convergence tolerance on function value (default 1e-4).
/// - Returns: Optimization result with converged flag.
func powellMinimize(
    objective: ([Double]) -> Double,
    x0: [Double],
    maxIter: Int = 600_000,
    ftol: Double = 1e-4
) -> OptimizeResult {
    let n = x0.count
    var x = x0
    var fval = objective(x)
    var totalNfev = 1

    // Direction set: start with identity (standard basis vectors).
    // Ported from scipy/_optimize.py:_minimize_powell
    var directions: [[Double]] = (0..<n).map { i in
        var d = [Double](repeating: 0.0, count: n)
        d[i] = 1.0
        return d
    }

    var converged = false

    for _ in 0..<maxIter {
        let fvalStart = fval
        let xStart = x

        var biggestDecrease = 0.0
        var biggestDecreaseIdx = 0

        // Minimize along each direction in the current set.
        // Line search tolerance matches SciPy: ftol * 100 (not the tighter Brent default).
        let lineTol = ftol * 100
        for i in 0..<n {
            let dir = directions[i]
            let fvalBefore = fval
            let (newX, newFval, evalCount) = lineMinimize(
                objective: objective,
                x: x,
                direction: dir,
                tol: lineTol
            )
            totalNfev += evalCount
            x = newX
            fval = newFval

            let decrease = fvalBefore - fval
            if decrease > biggestDecrease {
                biggestDecrease = decrease
                biggestDecreaseIdx = i
            }
        }

        // Check convergence: relative change in function value.
        // Ported from scipy/_optimize.py:_minimize_powell convergence check
        let tol = ftol * (abs(fvalStart) + abs(fval)) + 1e-20
        if 2.0 * (fvalStart - fval) <= tol {
            converged = true
            break
        }

        // Combined displacement for this iteration.
        let delta = vecSubtract(x, xStart)
        let deltaNorm = vecNorm(delta)
        if deltaNorm < 1e-12 {
            converged = true
            break
        }

        // Extrapolated point.
        let xe = vecAdd(x, delta)
        let fxe = objective(xe)
        totalNfev += 1

        // SciPy heuristic: update direction set unless extrapolation is not helpful.
        // Ported from scipy/_optimize.py:_minimize_powell direction update heuristic
        let f0 = fvalStart
        let fx = fval
        let t1 = (f0 - fx - biggestDecrease)
        let t2 = f0 - fxe
        if fxe < f0 && 2.0 * (f0 - 2.0 * fx + fxe) * t1 * t1 < biggestDecrease * t2 * t2 {
            let normDelta = vecNormalize(delta)
            let (newX2, newFval2, evalCount2) = lineMinimize(
                objective: objective,
                x: x,
                direction: normDelta,
                tol: lineTol
            )
            totalNfev += evalCount2
            // Always accept the line search result along the combined direction (matches SciPy).
            x = newX2
            fval = newFval2
            // Rotate: move largest-decrease direction to end, put combined direction last.
            // SciPy: direc[bigind] = direc[-1]; direc[-1] = direc1
            directions[biggestDecreaseIdx] = directions[n - 1]
            directions[n - 1] = normDelta
        }
    }

    return OptimizeResult(x: x, fun: fval, nfev: totalNfev, converged: converged)
}

// MARK: - 1D Line minimization

/// Minimizes f along a line from x in direction d, returning (new_x, new_fval, nfev).
private func lineMinimize(
    objective: ([Double]) -> Double,
    x: [Double],
    direction: [Double],
    tol: Double = 1.48e-8
) -> ([Double], Double, Int) {
    var nfev = 0
    let f1D: (Double) -> Double = { t in
        nfev += 1
        return objective(vecAdd(x, vecScale(direction, t)))
    }

    let (ax, bx, cx, _, _, _, bracketEvals) = mnbrak(f: f1D, ax: 0.0, bx: 1.0)
    nfev += bracketEvals

    let (tMin, fMin, brentEvals) = brentMinimize1D(f: f1D, ax: ax, bx: bx, cx: cx, tol: tol)
    nfev += brentEvals

    let newX = vecAdd(x, vecScale(direction, tMin))
    return (newX, fMin, nfev)
}

// MARK: - Brent's method for 1D minimization

/// Minimizes f in the bracket [ax, cx] using Brent's method.
///
/// Ported from scipy/_optimize.py:Brent / brent (1973).
///
/// - Parameters:
///   - f: 1D function to minimize.
///   - ax, bx, cx: Bracket such that f(bx) < f(ax) and f(bx) < f(cx).
///   - tol: Convergence tolerance.
///   - maxIter: Maximum iterations.
/// - Returns: (x, f(x), nfev)
func brentMinimize1D(
    f: (Double) -> Double,
    ax: Double,
    bx: Double,
    cx: Double,
    tol: Double = 1.48e-8,
    maxIter: Int = 500
) -> (x: Double, fx: Double, nfev: Int) {
    // Ported from scipy/_optimize.py:Brent.__call__ / brent
    let cgold = 0.3819660112501051  // (3 - sqrt(5)) / 2
    let zeps = 1.0e-10

    var nfev = 0

    var a = min(ax, cx)
    var b = max(ax, cx)

    var x = bx
    var w = bx
    var v = bx
    var fx = f(x); nfev += 1
    var fw = fx
    var fv = fx

    var d = 0.0
    var e = 0.0  // step size from two steps ago

    for _ in 0..<maxIter {
        let xm = 0.5 * (a + b)
        let tol1 = tol * abs(x) + zeps
        let tol2 = 2.0 * tol1

        // Test convergence.
        if abs(x - xm) <= tol2 - 0.5 * (b - a) {
            return (x, fx, nfev)
        }

        var newD: Double
        if abs(e) > tol1 {
            // Attempt inverse parabolic interpolation.
            let r = (x - w) * (fx - fv)
            let q0 = (x - v) * (fx - fw)
            var p = (x - v) * q0 - (x - w) * r
            var q = 2.0 * (q0 - r)
            if q > 0.0 { p = -p } else { q = -q }
            let ePrev = e
            e = d
            if abs(p) < abs(0.5 * q * ePrev) && p > q * (a - x) && p < q * (b - x) {
                // Parabolic step is acceptable.
                newD = p / q
                let u = x + newD
                if (u - a) < tol2 || (b - u) < tol2 {
                    newD = (x < xm) ? tol1 : -tol1
                }
            } else {
                // Golden section step.
                e = (x < xm) ? b - x : a - x
                newD = cgold * e
            }
        } else {
            // Golden section step.
            e = (x < xm) ? b - x : a - x
            newD = cgold * e
        }

        let u = x + (abs(newD) >= tol1 ? newD : (newD > 0 ? tol1 : -tol1))
        d = newD
        let fu = f(u); nfev += 1

        if fu <= fx {
            // u is better than x: shrink bracket and update best point.
            if u < x { b = x } else { a = x }
            v = w;  fv = fw
            w = x;  fw = fx
            x = u;  fx = fu
        } else {
            // u is worse: shrink bracket around u.
            if u < x { a = u } else { b = u }
            if fu <= fw || w == x {
                v = w;  fv = fw
                w = u;  fw = fu
            } else if fu <= fv || v == x || v == w {
                v = u;  fv = fu
            }
        }
    }

    return (x, fx, nfev)
}

// MARK: - Bracket finding (mnbrak)

/// Finds a downhill bracket [ax, bx, cx] for 1D minimization.
///
/// Ported from scipy/_optimize.py:bracket / Numerical Recipes mnbrak.
/// Returns (ax, bx, cx, fa, fb, fc, nfev).
private func mnbrak(
    f: (Double) -> Double,
    ax initialAx: Double,
    bx initialBx: Double
) -> (Double, Double, Double, Double, Double, Double, Int) {
    let gold = 1.618033988749895
    let glimit = 110.0
    let tiny = 1.0e-20

    var nfev = 0
    var ax = initialAx
    var bx = initialBx
    var fa = f(ax); nfev += 1
    var fb = f(bx); nfev += 1

    // Ensure we move downhill from a to b.
    if fb > fa {
        swap(&ax, &bx)
        swap(&fa, &fb)
    }

    var cx = bx + gold * (bx - ax)
    var fc = f(cx); nfev += 1

    while fb >= fc {
        let r = (bx - ax) * (fb - fc)
        let q = (bx - cx) * (fb - fa)
        let denom = max(abs(q - r), tiny) * ((q - r) >= 0 ? 2.0 : -2.0)
        var u = bx - ((bx - cx) * q - (bx - ax) * r) / denom
        let ulim = bx + glimit * (cx - bx)

        var fu: Double
        if (bx - u) * (u - cx) > 0.0 {
            fu = f(u); nfev += 1
            if fu < fc {
                ax = bx; fa = fb
                bx = u;  fb = fu
                return (ax, bx, cx, fa, fb, fc, nfev)
            } else if fu > fb {
                cx = u; fc = fu
                return (ax, bx, cx, fa, fb, fc, nfev)
            }
            u = cx + gold * (cx - bx)
            fu = f(u); nfev += 1
        } else if (cx - u) * (u - ulim) > 0.0 {
            fu = f(u); nfev += 1
            if fu < fc {
                bx = cx; fb = fc
                cx = u;  fc = fu
                u = cx + gold * (cx - bx)
                fu = f(u); nfev += 1
            }
        } else if (u - ulim) * (ulim - cx) >= 0.0 {
            u = ulim
            fu = f(u); nfev += 1
        } else {
            u = cx + gold * (cx - bx)
            fu = f(u); nfev += 1
        }

        ax = bx; fa = fb
        bx = cx; fb = fc
        cx = u;  fc = fu
    }

    return (ax, bx, cx, fa, fb, fc, nfev)
}

// MARK: - Vector helpers

private func vecAdd(_ a: [Double], _ b: [Double]) -> [Double] {
    zip(a, b).map { $0 + $1 }
}

private func vecSubtract(_ a: [Double], _ b: [Double]) -> [Double] {
    zip(a, b).map { $0 - $1 }
}

private func vecScale(_ a: [Double], _ s: Double) -> [Double] {
    a.map { $0 * s }
}

private func vecNorm(_ a: [Double]) -> Double {
    sqrt(a.reduce(0.0) { $0 + $1 * $1 })
}

private func vecNormalize(_ a: [Double]) -> [Double] {
    let n = vecNorm(a)
    guard n > 1e-12 else { return a }
    return vecScale(a, 1.0 / n)
}
