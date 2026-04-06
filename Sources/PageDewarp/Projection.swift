// Projection.swift
// Ported from src/page_dewarp/projection.py

import Foundation
import Accelerate
import simd
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
    var a = alpha + beta
    var b = -2 * alpha - beta
    var c = alpha

    guard !xyCoords.isEmpty else { return [] }

    let rvec = Array(pvec[DewarpConfig.rvecIdx])
    let tvec = Array(pvec[DewarpConfig.tvecIdx])
    var fLocal = DewarpConfig.focalLength

    // Rodrigues rotation (pure Swift, matches OpenCV cv::Rodrigues exactly)
    let (R, _) = rodrigues(rvec)
    let tx = tvec[0], ty = tvec[1], tz = tvec[2]

    let n = xyCoords.count
    let nD = vDSP_Length(n)

    // Extract x and y into flat contiguous arrays for vDSP
    var xCoords = [Double](repeating: 0.0, count: n)
    var yCoords = [Double](repeating: 0.0, count: n)
    for i in 0..<n {
        xCoords[i] = xyCoords[i][0]
        yCoords[i] = xyCoords[i][1]
    }

    // Cubic z-coords via Horner's method using vDSP — Ported from projection.py:46-49
    // z = ((cubicA*x + cubicB)*x + cubicC)*x
    var zCoords  = [Double](repeating: 0.0, count: n)
    var tempPoly = [Double](repeating: 0.0, count: n)
    vDSP_vsmsaD(xCoords, 1, &a, &b, &tempPoly, 1, nD)   // tempPoly = a*x + b
    vDSP_vmulD(tempPoly, 1, xCoords, 1, &tempPoly, 1, nD) // tempPoly = tempPoly * x
    vDSP_vsaddD(tempPoly, 1, &c, &tempPoly, 1, nD)         // tempPoly = tempPoly + c
    vDSP_vmulD(tempPoly, 1, xCoords, 1, &zCoords, 1, nD)  // z = tempPoly * x

    // Camera space: P_cam = R * [x, y, z]^T + t — Ported from projection.py:50-56
    var cxArr = [Double](repeating: 0.0, count: n)
    var cyArr = [Double](repeating: 0.0, count: n)
    var czArr = [Double](repeating: 0.0, count: n)
    var r0 = R[0], r1 = R[1], r2 = R[2]
    var r3 = R[3], r4 = R[4], r5 = R[5]
    var r6 = R[6], r7 = R[7], r8 = R[8]
    var txV = tx, tyV = ty, tzV = tz
    // cx = R[0]*x + R[1]*y + R[2]*z + tx
    vDSP_vsmulD(xCoords, 1, &r0, &cxArr, 1, nD)
    vDSP_vsmaD(yCoords, 1, &r1, cxArr, 1, &cxArr, 1, nD)
    vDSP_vsmaD(zCoords, 1, &r2, cxArr, 1, &cxArr, 1, nD)
    vDSP_vsaddD(cxArr, 1, &txV, &cxArr, 1, nD)
    // cy = R[3]*x + R[4]*y + R[5]*z + ty
    vDSP_vsmulD(xCoords, 1, &r3, &cyArr, 1, nD)
    vDSP_vsmaD(yCoords, 1, &r4, cyArr, 1, &cyArr, 1, nD)
    vDSP_vsmaD(zCoords, 1, &r5, cyArr, 1, &cyArr, 1, nD)
    vDSP_vsaddD(cyArr, 1, &tyV, &cyArr, 1, nD)
    // cz = R[6]*x + R[7]*y + R[8]*z + tz
    vDSP_vsmulD(xCoords, 1, &r6, &czArr, 1, nD)
    vDSP_vsmaD(yCoords, 1, &r7, czArr, 1, &czArr, 1, nD)
    vDSP_vsmaD(zCoords, 1, &r8, czArr, 1, &czArr, 1, nD)
    vDSP_vsaddD(czArr, 1, &tzV, &czArr, 1, nD)

    // iz = 1 / cz — Ported from projection.py:52
    var iz = [Double](repeating: 0.0, count: n)
    var one = 1.0
    vDSP_svdivD(&one, czArr, 1, &iz, 1, nD)

    // u = f * cx * iz, v = f * cy * iz — Ported from projection.py:53-56
    var uCoords = [Double](repeating: 0.0, count: n)
    var vCoords = [Double](repeating: 0.0, count: n)
    vDSP_vmulD(cxArr, 1, iz, 1, &uCoords, 1, nD)
    vDSP_vsmulD(uCoords, 1, &fLocal, &uCoords, 1, nD)
    vDSP_vmulD(cyArr, 1, iz, 1, &vCoords, 1, nD)
    vDSP_vsmulD(vCoords, 1, &fLocal, &vCoords, 1, nD)

    // Pack results back into [[Double]]
    var result = [[Double]](repeating: [0.0, 0.0], count: n)
    for i in 0..<n {
        result[i][0] = uCoords[i]
        result[i][1] = vCoords[i]
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
