// AnalyticalGradient.swift
// Computes the dewarp objective value AND its analytical gradient in a single pass.
// Chain rule: pvec → keypoint indexing → cubic z(x) → Rodrigues + pinhole → squared error.

import Foundation

/// Compute the objective value and its analytical gradient in one pass.
///
/// Replaces the pattern of `makeObjective()` + `finiteDifferenceGradient()`.
/// The gradient is exact, computed via chain rule through the full projection pipeline.
///
/// - Parameters:
///   - pvec: Current parameter vector [rvec(3), tvec(3), cubic(2), ycoords, xcoords...].
///   - dstpoints: Target 2D points to match.
///   - keypointIndex: Index table from makeKeypointIndex.
///   - shearCost: Penalty coefficient for rvec[0].
///   - focalLength: Camera focal length (1.2).
/// - Returns: (f: objective value, grad: gradient vector of length pvec.count)
func objectiveAndGradient(
    pvec: [Double],
    dstpoints: [[Double]],
    keypointIndex: [[Int]],
    shearCost: Double,
    focalLength: Double
) -> (f: Double, grad: [Double]) {
    let n = pvec.count
    var grad = [Double](repeating: 0.0, count: n)
    var f = 0.0

    // Extract rvec, tvec
    let rvec = [pvec[0], pvec[1], pvec[2]]
    let tvec = [pvec[3], pvec[4], pvec[5]]

    // Cubic coefficients with clamp
    let rawAlpha = pvec[6], rawBeta = pvec[7]
    let alpha = max(-0.5, min(0.5, rawAlpha))
    let beta  = max(-0.5, min(0.5, rawBeta))
    let alphaClamped = abs(rawAlpha) >= 0.5
    let betaClamped  = abs(rawBeta) >= 0.5

    let cubicA = alpha + beta
    let cubicB = -2 * alpha - beta
    let cubicC = alpha

    // Build 3D points from keypoint index + cubic polynomial
    let nPts = keypointIndex.count
    var points3DFlat = [Double](repeating: 0.0, count: nPts * 3)
    var xVals = [Double](repeating: 0.0, count: nPts)

    for k in 0..<nPts {
        let x: Double, y: Double
        if k == 0 {
            x = 0.0; y = 0.0
        } else {
            x = pvec[keypointIndex[k][0]]
            y = pvec[keypointIndex[k][1]]
        }
        let z = ((cubicA * x + cubicB) * x + cubicC) * x
        points3DFlat[k*3]   = x
        points3DFlat[k*3+1] = y
        points3DFlat[k*3+2] = z
        xVals[k] = x
    }

    // Compute Rodrigues rotation and its Jacobian (once for all points)
    let (R, dR_dr) = rodrigues(rvec)
    let tx = tvec[0], ty = tvec[1], tz = tvec[2]

    // Process each keypoint
    for k in 0..<dstpoints.count {
        let X = points3DFlat[k*3]
        let Y = points3DFlat[k*3+1]
        let Z = points3DFlat[k*3+2]

        // Camera space
        let cx = R[0]*X + R[1]*Y + R[2]*Z + tx
        let cy = R[3]*X + R[4]*Y + R[5]*Z + ty
        let cz = R[6]*X + R[7]*Y + R[8]*Z + tz

        // Projection
        let iz = 1.0 / cz
        let u = focalLength * cx * iz
        let v = focalLength * cy * iz

        // Squared error
        let du = u - dstpoints[k][0]
        let dv = v - dstpoints[k][1]
        f += du * du + dv * dv

        // Error gradient: 2·(proj - dst)
        let eu = 2.0 * du
        let ev = 2.0 * dv

        // Perspective Jacobian components
        let f_iz = focalLength * iz
        let f_iz2 = focalLength * iz * iz
        let J00 = f_iz
        let J02 = -f_iz2 * cx
        let J11 = f_iz
        let J12 = -f_iz2 * cy

        // --- rvec (indices 0–2) ---
        for m in 0..<3 {
            let dcx = dR_dr[0*3+m]*X + dR_dr[1*3+m]*Y + dR_dr[2*3+m]*Z
            let dcy = dR_dr[3*3+m]*X + dR_dr[4*3+m]*Y + dR_dr[5*3+m]*Z
            let dcz = dR_dr[6*3+m]*X + dR_dr[7*3+m]*Y + dR_dr[8*3+m]*Z
            let du_dr = J00*dcx + J02*dcz
            let dv_dr = J11*dcy + J12*dcz
            grad[m] += eu*du_dr + ev*dv_dr
        }

        // --- tvec (indices 3–5) ---
        grad[3] += eu * J00           // du/dtx
        grad[4] += ev * J11           // dv/dty
        grad[5] += eu * J02 + ev * J12  // d(u,v)/dtz

        // --- cubic α, β (indices 6–7) ---
        // dP_cam/dz uses R's third column: (R[2], R[5], R[8])
        let du_dz = J00*R[2] + J02*R[8]
        let dv_dz = J11*R[5] + J12*R[8]
        let errDotDz = eu*du_dz + ev*dv_dz

        let x = xVals[k]
        let x2 = x * x, x3 = x2 * x

        if !alphaClamped {
            let dz_dalpha = x3 - 2*x2 + x
            grad[6] += errDotDz * dz_dalpha
        }
        if !betaClamped {
            let dz_dbeta = x3 - x2
            grad[7] += errDotDz * dz_dbeta
        }

        // Skip keypoint 0 (origin — fixed coordinates, no gradient)
        guard k > 0 else { continue }

        let xIdx = keypointIndex[k][0]
        let yIdx = keypointIndex[k][1]

        // --- ycoord ---
        // dP_cam/dy uses R's second column: (R[1], R[4], R[7])
        let du_dy = J00*R[1] + J02*R[7]
        let dv_dy = J11*R[4] + J12*R[7]
        grad[yIdx] += eu*du_dy + ev*dv_dy

        // --- xcoord ---
        // Two paths: direct via x, and indirect via z(x)
        let dz_dx = 3*cubicA*x2 - 2*(2*alpha + beta)*x + alpha
        // dP_cam/dx = R col0 + R col2 · dz/dx
        let du_dx = J00*(R[0] + R[2]*dz_dx) + J02*(R[6] + R[8]*dz_dx)
        let dv_dx = J11*(R[3] + R[5]*dz_dx) + J12*(R[6] + R[8]*dz_dx)
        grad[xIdx] += eu*du_dx + ev*dv_dx
    }

    // Shear penalty
    if shearCost > 0 {
        f += shearCost * pvec[0] * pvec[0]
        grad[0] += 2.0 * shearCost * pvec[0]
    }

    return (f, grad)
}
