// Objective.swift
// Ported from src/page_dewarp/optimise/_base.py

import Foundation

/// Creates the optimization objective function for page dewarping.
///
/// Fused-loop implementation: computes gather→cubic→rotate→project→error per
/// point in a single pass, keeping all intermediate values in registers and
/// L1 cache. Eliminates the vDSP function call overhead (~15 calls × dispatch
/// cost) that was dominant for the small point count (~179 pts).
///
/// Pre-allocates dstX/dstY/xPvecIdx/yPvecIdx once at closure creation.
///
/// Returns a closure that computes the sum of squared projection errors between
/// target dstpoints and the projected keypoints at a given parameter vector.
/// Optionally adds a shear penalty proportional to pvec[rvecRange.lowerBound]^2.
///
/// Ported from optimise/_base.py:34-64
///
/// - Parameters:
///   - dstpoints: Target 2D points to match (one per keypoint row).
///   - keypointIndex: Index table from makeKeypointIndex.
///   - shearCost: Penalty coefficient for rotation shear (0.0 = no penalty).
///   - rvecRange: Range of rvec indices in pvec (typically 0..<3).
/// - Returns: A closure `([Double]) -> Double` suitable for Powell/L-BFGS-B minimization.
func makeObjective(
    dstpoints: [[Double]],
    keypointIndex: [[Int]],
    shearCost: Double,
    rvecRange: Range<Int>
) -> ([Double]) -> Double {
    let n = dstpoints.count

    // Pre-flatten destination points into contiguous arrays (allocated once).
    var dstX = [Double](repeating: 0.0, count: n)
    var dstY = [Double](repeating: 0.0, count: n)
    for i in 0..<n {
        dstX[i] = dstpoints[i][0]
        dstY[i] = dstpoints[i][1]
    }

    // Pre-extract pvec indices from keypointIndex (allocated once).
    // keypointIndex[0] = origin — x/y are fixed at 0.0, its indices are unused.
    var xPvecIdx = [Int](repeating: 0, count: n)
    var yPvecIdx = [Int](repeating: 0, count: n)
    for i in 1..<n {
        xPvecIdx[i] = keypointIndex[i][0]
        yPvecIdx[i] = keypointIndex[i][1]
    }

    return { pvec in
        // Cubic coefficients with clamping — Ported from projection.py:36-47
        let alpha = max(-0.5, min(0.5, pvec[6]))
        let beta  = max(-0.5, min(0.5, pvec[7]))
        let cubicA = alpha + beta
        let cubicB = -2.0 * alpha - beta
        let cubicC = alpha

        // Rodrigues rotation matrix (once per call, no Jacobian) — Ported from projection.py:50-56
        let R = rodriguesRotationOnly([pvec[0], pvec[1], pvec[2]])
        let r0 = R[0], r1 = R[1], r2 = R[2]
        let r3 = R[3], r4 = R[4], r5 = R[5]
        let r6 = R[6], r7 = R[7], r8 = R[8]
        let tx = pvec[3], ty = pvec[4], tz = pvec[5]
        let f = DewarpConfig.focalLength

        // Origin point (i=0): x=y=z=0, so P_cam = t — Ported from keypoints.py:62
        var error: Double
        do {
            let cz = tz
            let iz = 1.0 / cz
            let du = f * tx * iz - dstX[0]
            let dv = f * ty * iz - dstY[0]
            error = du*du + dv*dv
        }

        // Fused gather→cubic→rotate→project→error per point — Ported from projection.py:46-56 and optimise/_base.py:55-56
        for i in 1..<n {
            let x = pvec[xPvecIdx[i]]
            let y = pvec[yPvecIdx[i]]
            // z = ((cubicA*x + cubicB)*x + cubicC)*x  (Horner's method)
            let z = ((cubicA*x + cubicB)*x + cubicC)*x
            // Camera space: P_cam = R·[x,y,z]ᵀ + t
            let cx = r0*x + r1*y + r2*z + tx
            let cy = r3*x + r4*y + r5*z + ty
            let cz = r6*x + r7*y + r8*z + tz
            let iz = 1.0 / cz
            // Projection: u = f·cx/cz, v = f·cy/cz
            let du = f * cx * iz - dstX[i]
            let dv = f * cy * iz - dstY[i]
            error += du*du + dv*dv
        }

        // Shear penalty — Ported from optimise/_base.py:57-60
        if shearCost > 0.0 {
            let rx = pvec[rvecRange.lowerBound]
            error += shearCost * rx * rx
        }
        return error
    }
}
