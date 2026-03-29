// Keypoints.swift
// Ported from src/page_dewarp/keypoints.py

import Foundation

/// Builds a 2D integer index mapping keypoints to parameter-vector positions.
///
/// Returns an array of (npts+1) pairs [pointIdx, spanIdx], where:
/// - Row 0 is [0, 0] (reserved origin row)
/// - Rows 1..npts: column 0 = 8 + nspans + (row-1), column 1 = 8 + spanIndex
///
/// The offset of 8 arises from the fixed head of pvec: [rvec(3), tvec(3), cubic(2)].
///
/// Ported from keypoints.py:17-43
///
/// - Parameter spanCounts: Number of sampled keypoints per span.
/// - Returns: Array of shape (npts+1, 2) encoding pvec index pairs.
func makeKeypointIndex(spanCounts: [Int]) -> [[Int]] {
    let nspans = spanCounts.count
    let npts = spanCounts.reduce(0, +)
    // Ported from keypoints.py:36-42
    var index = Array(repeating: [0, 0], count: npts + 1)
    var start = 1
    for (i, count) in spanCounts.enumerated() {
        let end = start + count
        for j in start..<end {
            index[j][1] = 8 + i
        }
        start = end
    }
    for j in 1...npts {
        index[j][0] = (j - 1) + 8 + nspans
    }
    return index
}

/// Projects keypoints encoded in a parameter vector into 2D image coordinates.
///
/// Extracts (x, y) pairs from pvec using keypointIndex fancy-indexing,
/// forces the first row to the origin [0, 0], then calls projectXY.
///
/// Python fancy indexing: pvec[keypointIndex] extracts rows where
/// xyCoords[k] = [pvec[keypointIndex[k][0]], pvec[keypointIndex[k][1]]].
///
/// Ported from keypoints.py:46-63
///
/// - Parameters:
///   - pvec: Full parameter vector [rvec(3), tvec(3), cubic(2), ycoords, xcoords...].
///   - keypointIndex: Index table from makeKeypointIndex.
/// - Returns: Projected 2D points as [[x, y]] array.
func projectKeypoints(pvec: [Double], keypointIndex: [[Int]]) -> [[Double]] {
    // Ported from keypoints.py:61-62
    var xyCoords: [[Double]] = keypointIndex.map { idx in
        [pvec[idx[0]], pvec[idx[1]]]
    }
    xyCoords[0] = [0.0, 0.0]  // first row is origin, ported from keypoints.py:62
    return projectXY(xyCoords: xyCoords, pvec: pvec)
}
