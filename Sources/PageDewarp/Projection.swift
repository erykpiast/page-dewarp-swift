// Projection.swift
// Ported from src/page_dewarp/projection.py

import Foundation
import Accelerate

/// Projects normalized (x, y) coordinates through a cubic warp surface model into image space.
///
/// Pure-Swift implementation — no ObjC bridge, uses rodrigues() directly.
/// No distortion (K = diag(f, f, 1)).
///
/// Used in the optimization hot loop (projectKeypoints) and by the Remapper.
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

/// Bulk-project N normalized (x, y) coordinates through the cubic warp model.
///
/// Uses flat arrays to eliminate per-point heap allocations. Applies projection and
/// norm2pix in a single pass, returning Float pixel maps ready for OpenCV remap.
///
/// Ported from projection.py:19-57 + normalisation.py:33-51
///
/// - Parameters:
///   - xs: Flat array of N normalized x coordinates.
///   - ys: Flat array of N normalized y coordinates.
///   - pvec: The full parameter vector — rvec at [0..<3], tvec at [3..<6], cubic at [6..<8].
///   - shape: Source image (height, width) for norm2pix scaling.
/// - Returns: Flat Float arrays (mapX, mapY) of pixel coordinates suitable for cv::remap.
func projectXYBulk(
    xs: [Double],
    ys: [Double],
    pvec: [Double],
    shape: (height: Int, width: Int)
) -> (mapX: [Float], mapY: [Float]) {
    let n = xs.count
    guard n > 0 else { return ([], []) }

    // Cubic polynomial coefficients — Ported from projection.py:36-47
    let alpha = max(-0.5, min(0.5, pvec[DewarpConfig.cubicIdx.lowerBound]))
    let beta  = max(-0.5, min(0.5, pvec[DewarpConfig.cubicIdx.lowerBound + 1]))
    let a = alpha + beta
    let b = -2 * alpha - beta
    let c = alpha

    let rvec = Array(pvec[DewarpConfig.rvecIdx])
    let tvec = Array(pvec[DewarpConfig.tvecIdx])
    let R  = rodriguesRotationOnly(rvec)
    let tx = tvec[0], ty = tvec[1], tz = tvec[2]

    // norm2pix scale/offset — Ported from normalisation.py:39-43
    let scl     = Double(max(shape.height, shape.width)) * 0.5
    let offsetX = 0.5 * Double(shape.width)
    let offsetY = 0.5 * Double(shape.height)

    let vn = vDSP_Length(n)

    // Mutable scalars required by vDSP (takes inout pointer)
    var coefA = a, coefB = b, coefC = c
    var r0 = R[0], r1 = R[1], r2 = R[2]
    var r3 = R[3], r4 = R[4], r5 = R[5]
    var r6 = R[6], r7 = R[7], r8 = R[8]
    var f  = DewarpConfig.focalLength

    // ── Step 1: Horner polynomial zs[i] = ((a·xs[i] + b)·xs[i] + c)·xs[i] ──
    // Ported from projection.py:46-49
    var zs  = [Double](repeating: 0.0, count: n)
    var tmp = [Double](repeating: 0.0, count: n)
    vDSP_vsmulD(xs, 1, &coefA, &zs, 1, vn)      // zs = a*xs
    vDSP_vsaddD(zs, 1, &coefB, &zs, 1, vn)       // zs += b
    vDSP_vmulD( zs, 1, xs, 1, &zs, 1, vn)        // zs *= xs
    vDSP_vsaddD(zs, 1, &coefC, &zs, 1, vn)       // zs += c
    vDSP_vmulD( zs, 1, xs, 1, &zs, 1, vn)        // zs *= xs

    // ── Step 2: Camera-space linear combinations ──
    // cx[i] = R[0]*xs[i] + R[1]*ys[i] + R[2]*zs[i] + tx
    // Ported from projection.py:50-52
    var cx = [Double](repeating: tx, count: n)
    var cy = [Double](repeating: ty, count: n)
    var cz = [Double](repeating: tz, count: n)

    vDSP_vsmulD(xs, 1, &r0, &tmp, 1, vn); vDSP_vaddD(cx, 1, tmp, 1, &cx, 1, vn)
    vDSP_vsmulD(ys, 1, &r1, &tmp, 1, vn); vDSP_vaddD(cx, 1, tmp, 1, &cx, 1, vn)
    vDSP_vsmulD(zs, 1, &r2, &tmp, 1, vn); vDSP_vaddD(cx, 1, tmp, 1, &cx, 1, vn)

    vDSP_vsmulD(xs, 1, &r3, &tmp, 1, vn); vDSP_vaddD(cy, 1, tmp, 1, &cy, 1, vn)
    vDSP_vsmulD(ys, 1, &r4, &tmp, 1, vn); vDSP_vaddD(cy, 1, tmp, 1, &cy, 1, vn)
    vDSP_vsmulD(zs, 1, &r5, &tmp, 1, vn); vDSP_vaddD(cy, 1, tmp, 1, &cy, 1, vn)

    vDSP_vsmulD(xs, 1, &r6, &tmp, 1, vn); vDSP_vaddD(cz, 1, tmp, 1, &cz, 1, vn)
    vDSP_vsmulD(ys, 1, &r7, &tmp, 1, vn); vDSP_vaddD(cz, 1, tmp, 1, &cz, 1, vn)
    vDSP_vsmulD(zs, 1, &r8, &tmp, 1, vn); vDSP_vaddD(cz, 1, tmp, 1, &cz, 1, vn)

    // ── Step 3: Perspective division us = f*cx/cz, vs = f*cy/cz ──
    // Ported from projection.py:53-56
    // Note: vDSP_vdivD(A,B,D) computes D[n] = B[n]/A[n] — A is the *divisor*
    var us = [Double](repeating: 0.0, count: n)
    var vs = [Double](repeating: 0.0, count: n)
    vDSP_vsmulD(cx, 1, &f, &us, 1, vn)       // us = f*cx
    vDSP_vdivD( cz, 1, us, 1, &us, 1, vn)    // us = f*cx / cz
    vDSP_vsmulD(cy, 1, &f, &vs, 1, vn)       // vs = f*cy
    vDSP_vdivD( cz, 1, vs, 1, &vs, 1, vn)    // vs = f*cy / cz

    // ── Step 4: norm2pix — pixX = u*scl + offsetX  (u already has f from step 3) ──
    // Ported from normalisation.py:43-44
    var sclD = scl, offsetXD = offsetX, offsetYD = offsetY
    vDSP_vsmulD(us, 1, &sclD, &us, 1, vn)         // us = u*scl
    vDSP_vsaddD(us, 1, &offsetXD, &us, 1, vn)     // us += offsetX
    vDSP_vsmulD(vs, 1, &sclD, &vs, 1, vn)         // vs = v*scl
    vDSP_vsaddD(vs, 1, &offsetYD, &vs, 1, vn)     // vs += offsetY

    // ── Step 5: Double → Float conversion for cv::remap ──
    var mapX = [Float](repeating: 0, count: n)
    var mapY = [Float](repeating: 0, count: n)
    vDSP_vdpsp(us, 1, &mapX, 1, vn)
    vDSP_vdpsp(vs, 1, &mapY, 1, vn)

    return (mapX, mapY)
}

