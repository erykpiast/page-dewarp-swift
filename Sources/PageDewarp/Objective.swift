// Objective.swift
// Ported from src/page_dewarp/optimise/_base.py

import Foundation
import Accelerate

/// Creates the optimization objective function for page dewarping.
///
/// Vectorized implementation using Apple Accelerate (vDSP) for improved performance.
/// Pre-allocates working buffers once at closure creation to eliminate per-call heap
/// allocations. Uses flat contiguous [Double] arrays internally.
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
    let nD = vDSP_Length(n)

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

    // Pre-allocated working buffers — reused on every call (zero per-call heap allocation).
    var xCoords  = [Double](repeating: 0.0, count: n)
    var yCoords  = [Double](repeating: 0.0, count: n)
    var zCoords  = [Double](repeating: 0.0, count: n)
    var cx       = [Double](repeating: 0.0, count: n)
    var cy       = [Double](repeating: 0.0, count: n)
    var cz       = [Double](repeating: 0.0, count: n)
    var iz       = [Double](repeating: 0.0, count: n)
    var uCoords  = [Double](repeating: 0.0, count: n)
    var vCoords  = [Double](repeating: 0.0, count: n)
    var du       = [Double](repeating: 0.0, count: n)
    var dv       = [Double](repeating: 0.0, count: n)
    var tempPoly = [Double](repeating: 0.0, count: n)

    return { pvec in
        // Cubic coefficients with clamping — Ported from projection.py:36-47
        let alpha = max(-0.5, min(0.5, pvec[6]))
        let beta  = max(-0.5, min(0.5, pvec[7]))
        var cubicA = alpha + beta
        var cubicB = -2.0 * alpha - beta
        var cubicC = alpha

        // Rodrigues rotation matrix (once per call) — Ported from projection.py:50-56
        let (R, _) = rodrigues([pvec[0], pvec[1], pvec[2]])
        let tx = pvec[3], ty = pvec[4], tz = pvec[5]
        var fLocal = DewarpConfig.focalLength

        // Gather (x, y) from pvec using pre-extracted indices.
        // Row 0 = origin [0, 0] — Ported from keypoints.py:62
        xCoords[0] = 0.0
        yCoords[0] = 0.0
        for i in 1..<n {
            xCoords[i] = pvec[xPvecIdx[i]]
            yCoords[i] = pvec[yPvecIdx[i]]
        }

        // Cubic z-coords via Horner's method — Ported from projection.py:46-49
        // z = ((cubicA*x + cubicB)*x + cubicC)*x
        // Step 1: tempPoly = cubicA * x + cubicB
        vDSP_vsmsaD(xCoords, 1, &cubicA, &cubicB, &tempPoly, 1, nD)
        // Step 2: tempPoly = tempPoly * x
        vDSP_vmulD(tempPoly, 1, xCoords, 1, &tempPoly, 1, nD)
        // Step 3: tempPoly = tempPoly + cubicC
        vDSP_vsaddD(tempPoly, 1, &cubicC, &tempPoly, 1, nD)
        // Step 4: zCoords = tempPoly * x
        vDSP_vmulD(tempPoly, 1, xCoords, 1, &zCoords, 1, nD)

        // Camera space: P_cam = R * [x, y, z]^T + t — Ported from projection.py:50-56
        // Vectorized with vDSP: eliminates per-point scalar loop.
        // vDSP_vsmaD(A, strideA, &b_scalar, C, strideC, D, strideD, N): D[i] = A[i]*b + C[i]
        var r0 = R[0], r1 = R[1], r2 = R[2]
        var r3 = R[3], r4 = R[4], r5 = R[5]
        var r6 = R[6], r7 = R[7], r8 = R[8]
        var txV = tx, tyV = ty, tzV = tz
        // cx = R[0]*x + R[1]*y + R[2]*z + tx
        vDSP_vsmulD(xCoords, 1, &r0, &cx, 1, nD)
        vDSP_vsmaD(yCoords, 1, &r1, cx, 1, &cx, 1, nD)
        vDSP_vsmaD(zCoords, 1, &r2, cx, 1, &cx, 1, nD)
        vDSP_vsaddD(cx, 1, &txV, &cx, 1, nD)
        // cy = R[3]*x + R[4]*y + R[5]*z + ty
        vDSP_vsmulD(xCoords, 1, &r3, &cy, 1, nD)
        vDSP_vsmaD(yCoords, 1, &r4, cy, 1, &cy, 1, nD)
        vDSP_vsmaD(zCoords, 1, &r5, cy, 1, &cy, 1, nD)
        vDSP_vsaddD(cy, 1, &tyV, &cy, 1, nD)
        // cz = R[6]*x + R[7]*y + R[8]*z + tz
        vDSP_vsmulD(xCoords, 1, &r6, &cz, 1, nD)
        vDSP_vsmaD(yCoords, 1, &r7, cz, 1, &cz, 1, nD)
        vDSP_vsmaD(zCoords, 1, &r8, cz, 1, &cz, 1, nD)
        vDSP_vsaddD(cz, 1, &tzV, &cz, 1, nD)

        // iz = 1 / cz — Ported from projection.py:52
        // vDSP_svdivD(scalar, vector, stride, output, stride, n): output[i] = scalar / vector[i]
        var one = 1.0
        vDSP_svdivD(&one, cz, 1, &iz, 1, nD)  // iz[i] = 1.0 / cz[i]

        // u = focalLength * cx * iz, v = focalLength * cy * iz — Ported from projection.py:53-56
        vDSP_vmulD(cx, 1, iz, 1, &uCoords, 1, nD)
        vDSP_vsmulD(uCoords, 1, &fLocal, &uCoords, 1, nD)
        vDSP_vmulD(cy, 1, iz, 1, &vCoords, 1, nD)
        vDSP_vsmulD(vCoords, 1, &fLocal, &vCoords, 1, nD)

        // Squared error: sum((u - dstX)^2 + (v - dstY)^2) — Ported from optimise/_base.py:55-56
        // vDSP_vsubD: C[i] = B[i] - A[i]
        vDSP_vsubD(dstX, 1, uCoords, 1, &du, 1, nD)  // du = uCoords - dstX
        vDSP_vsubD(dstY, 1, vCoords, 1, &dv, 1, nD)  // dv = vCoords - dstY
        var errU = 0.0, errV = 0.0
        vDSP_dotprD(du, 1, du, 1, &errU, nD)
        vDSP_dotprD(dv, 1, dv, 1, &errV, nD)
        var error = errU + errV

        // Shear penalty — Ported from optimise/_base.py:57-60
        if shearCost > 0.0 {
            let rx = pvec[rvecRange.lowerBound]
            error += shearCost * rx * rx
        }
        return error
    }
}
