// Projection.swift
// Ported from src/page_dewarp/projection.py

import Foundation
#if SWIFT_PACKAGE
import OpenCVBridge
#endif

/// Projects normalized (x, y) coordinates through a cubic warp surface model into image space.
///
/// Pure-Swift equivalent of projectXY — no ObjC bridge, uses rodrigues() directly.
/// Produces identical results to projectXY (no distortion, K = diag(f, f, 1)).
///
/// Used in the optimization hot loop (projectKeypoints). The Remapper still uses
/// the OpenCV-backed projectXY.
///
/// Ported from projection.py:19-57
///
/// - Parameters:
///   - xyCoords: An (N, 2) array of normalized (x, y) points.
///   - pvec: The full parameter vector — rvec at [0..<3], tvec at [3..<6], cubic at [6..<8].
/// - Returns: An (N, 2) array of projected 2D image points.
func projectXYPure(xyCoords: [[Double]], pvec: [Double]) -> [[Double]] {
    // Ported from projection.py:36-42
    let alpha = max(-0.5, min(0.5, pvec[DewarpConfig.cubicIdx.lowerBound]))
    let beta = max(-0.5, min(0.5, pvec[DewarpConfig.cubicIdx.lowerBound + 1]))

    // Cubic polynomial coefficients: poly = [alpha+beta, -2*alpha-beta, alpha, 0]
    // Ported from projection.py:44-47
    let a = alpha + beta
    let b = -2 * alpha - beta
    let c = alpha

    guard !xyCoords.isEmpty else { return [] }

    let rvec = Array(pvec[DewarpConfig.rvecIdx])
    let tvec = Array(pvec[DewarpConfig.tvecIdx])
    let f = DewarpConfig.focalLength

    // Rodrigues rotation (pure Swift, matches OpenCV cv::Rodrigues exactly)
    let (R, _) = rodrigues(rvec)
    let tx = tvec[0], ty = tvec[1], tz = tvec[2]

    var result = [[Double]](repeating: [0.0, 0.0], count: xyCoords.count)
    for i in 0..<xyCoords.count {
        let x = xyCoords[i][0]
        let y = xyCoords[i][1]
        // Cubic z from polynomial, d=0 — Ported from projection.py:46-49
        let z = ((a * x + b) * x + c) * x

        // Rotate + translate: P_cam = R * [x, y, z]^T + t
        let cx = R[0]*x + R[1]*y + R[2]*z + tx
        let cy = R[3]*x + R[4]*y + R[5]*z + ty
        let cz = R[6]*x + R[7]*y + R[8]*z + tz

        // Pinhole projection: K = diag(f, f, 1), no distortion, no principal point offset
        // Ported from projection.py:50-56
        let iz = 1.0 / cz
        result[i][0] = f * cx * iz
        result[i][1] = f * cy * iz
    }
    return result
}

/// Projects normalized (x, y) coordinates through a cubic warp surface model into image space.
///
/// Builds a cubic polynomial z(x) = ((a·x + b)·x + c)·x from pvec's cubic coefficients,
/// then calls OpenCV projectPoints to map (x, y, z) 3D points into 2D image coordinates.
///
/// Ported from projection.py:19-57
///
/// - Parameters:
///   - xyCoords: An (N, 2) array of normalized (x, y) points.
///   - pvec: The full parameter vector — rvec at [0..<3], tvec at [3..<6], cubic at [6..<8].
/// - Returns: An (N, 2) array of projected 2D image points.
func projectXY(xyCoords: [[Double]], pvec: [Double]) -> [[Double]] {
    // Ported from projection.py:36-42
    let alpha = max(-0.5, min(0.5, pvec[DewarpConfig.cubicIdx.lowerBound]))
    let beta = max(-0.5, min(0.5, pvec[DewarpConfig.cubicIdx.lowerBound + 1]))

    // Cubic polynomial coefficients: poly = [alpha+beta, -2*alpha-beta, alpha, 0]
    // np.polyval([a, b, c, d], x) = a*x^3 + b*x^2 + c*x + d
    // Implement via Horner's method: ((a*x + b)*x + c)*x + d, d=0
    // Ported from projection.py:44-47
    let a = alpha + beta
    let b = -2 * alpha - beta
    let c = alpha
    // d = 0 (implicit)

    // Build flat array of 3D object points [x0,y0,z0, x1,y1,z1, ...]
    // z is computed from the cubic polynomial evaluated at x
    // Ported from projection.py:46-49
    guard !xyCoords.isEmpty else { return [] }
    var points3DFlat: [NSNumber] = []
    for xy in xyCoords {
        let x = xy[0]
        let z = ((a * x + b) * x + c) * x  // Horner's method, d=0
        points3DFlat.append(NSNumber(value: xy[0]))
        points3DFlat.append(NSNumber(value: xy[1]))
        points3DFlat.append(NSNumber(value: z))
    }

    // Extract rvec, tvec from pvec; pass as Double (bridge uses doubleValue + CV_64F)
    // Ported from projection.py:50-56
    let rvec = Array(pvec[DewarpConfig.rvecIdx]).map { NSNumber(value: $0) }
    let tvec = Array(pvec[DewarpConfig.tvecIdx]).map { NSNumber(value: $0) }
    let kFlat = cameraMatrix().flatMap { $0 }.map { NSNumber(value: $0) }
    let distCoeffs: [NSNumber] = [0, 0, 0, 0, 0]

    // Project via OpenCV; bridge returns NSArray of CGPoint NSValues
    let projected = OpenCVWrapper.projectPointsWith3DPoints(
        points3DFlat,
        rvec: rvec,
        tvec: tvec,
        cameraMatrix: kFlat,
        distCoeffs: distCoeffs
    )

    return projected.map { val in
        let pt = val.cgPointValue
        return [Double(pt.x), Double(pt.y)]
    }
}
