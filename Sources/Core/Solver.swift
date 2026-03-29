// Solver.swift
// Ported from src/page_dewarp/solve.py

import Foundation

/// Errors produced by the solver.
enum SolverError: Error {
    case solvePnPFailed
}

/// Assembles an initial parameter vector for page flattening.
///
/// Uses four corner correspondences to estimate rotation/translation via solvePnP,
/// then concatenates rvec, tvec, cubic slopes, ycoords, and per-span xcoords.
///
/// Ported from solve.py:19-63
///
/// - Parameters:
///   - corners: 4 image-space 2D points (each [x, y]) — page corners in normalized coords.
///   - ycoords: One average y-position per span.
///   - xcoords: Per-span arrays of x-positions for sampled keypoints.
/// - Returns: `.success((pageDims, spanCounts, params))` or `.failure(.solvePnPFailed)`.
func getDefaultParams(
    corners: [[Double]],
    ycoords: [Double],
    xcoords: [[Double]]
) -> Result<(pageDims: (Double, Double), spanCounts: [Int], params: [Double]), SolverError> {
    // Compute page dimensions from corner distances.
    // Ported from solve.py:38
    let pageWidth = euclideanNorm(corners[1], corners[0])
    let pageHeight = euclideanNorm(corners[3], corners[0])

    // Build 3D object points of a flat page: TL, TR, BR, BL.
    // Ported from solve.py:42-49
    let objectPoints: [NSNumber] = [
        0, 0, 0,
        NSNumber(value: pageWidth), 0, 0,
        NSNumber(value: pageWidth), NSNumber(value: pageHeight), 0,
        0, NSNumber(value: pageHeight), 0,
    ]

    // Build 2D image points (corners in normalized image coordinates).
    // Ported from solve.py:51
    let imagePoints: [NSNumber] = corners.flatMap { pt in
        [NSNumber(value: pt[0]), NSNumber(value: pt[1])]
    }

    // Camera matrix (focal_length = 1.2).
    let kFlat = cameraMatrix().flatMap { $0 }.map { NSNumber(value: $0) }
    let distCoeffs: [NSNumber] = [0, 0, 0, 0, 0]

    // Solve PnP — estimate rotation and translation.
    // BUG FIX vs Python: check success flag (Python ignores it).
    // Ported from solve.py:51
    let pnpResult = OpenCVWrapper.solvePnP(
        withObjectPoints: objectPoints,
        imagePoints: imagePoints,
        cameraMatrix: kFlat,
        distCoeffs: distCoeffs
    )

    guard let success = pnpResult["success"] as? NSNumber, success.boolValue else {
        return .failure(.solvePnPFailed)
    }

    let rvecArr = (pnpResult["rvec"] as! [NSNumber]).map { $0.doubleValue }
    let tvecArr = (pnpResult["tvec"] as! [NSNumber]).map { $0.doubleValue }

    // Assemble parameter vector: [rvec(3), tvec(3), cubic(2), ycoords(N), xcoords(flat)].
    // Ported from solve.py:53-62
    let spanCounts = xcoords.map { $0.count }
    var params: [Double] = []
    params.append(contentsOf: rvecArr)         // indices 0..<3
    params.append(contentsOf: tvecArr)         // indices 3..<6
    params.append(contentsOf: [0.0, 0.0])     // cubic slopes, indices 6..<8
    params.append(contentsOf: ycoords)         // one per span, indices 8..<8+N
    for xc in xcoords {                        // xcoords flattened, indices 8+N..<end
        params.append(contentsOf: xc)
    }

    return .success((pageDims: (pageWidth, pageHeight), spanCounts: spanCounts, params: params))
}

// MARK: - Private helpers

/// Euclidean distance between two 2D points.
private func euclideanNorm(_ a: [Double], _ b: [Double]) -> Double {
    let dx = a[0] - b[0]
    let dy = a[1] - b[1]
    return sqrt(dx * dx + dy * dy)
}
