// Objective.swift
// Ported from src/page_dewarp/optimise/_base.py

import Foundation

/// Creates the optimization objective function for page dewarping.
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
/// - Returns: A closure `([Double]) -> Double` suitable for Powell minimization.
func makeObjective(
    dstpoints: [[Double]],
    keypointIndex: [[Int]],
    shearCost: Double,
    rvecRange: Range<Int>
) -> ([Double]) -> Double {
    return { pvec in
        // Project keypoints at current pvec and compute squared error
        // Ported from optimise/_base.py:55-56
        let projPts = projectKeypoints(pvec: pvec, keypointIndex: keypointIndex)
        var error = 0.0
        for i in 0..<dstpoints.count {
            let dx = dstpoints[i][0] - projPts[i][0]
            let dy = dstpoints[i][1] - projPts[i][1]
            error += dx * dx + dy * dy
        }
        // Shear penalty: penalise rotation around x-axis (pvec[rvecRange.lowerBound])
        // Ported from optimise/_base.py:57-60
        if shearCost > 0.0 {
            let rx = pvec[rvecRange.lowerBound]
            error += shearCost * rx * rx
        }
        return error
    }
}
