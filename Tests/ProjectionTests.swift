// ProjectionTests.swift
// Tests for projectXYPure() — pure-Swift projection, ported from projection.py

import XCTest
@testable import PageDewarp

final class ProjectionTests: XCTestCase {

    // Golden pvec extracted from golden_output/initial_params.json:
    //   rvec[0..3]  + tvec[3..6] + cubic[6..8]
    // All subsequent elements (ycoords, xcoords) are appended but unused by projectXYPure (only uses 0..<8).
    private let goldenPvec: [Double] = [
        // rvec (indices 0..<3)
        -0.0, 0.0, 0.0060951,
        // tvec (indices 3..<6)
        -0.59620073, -0.94444305, 1.20000005,
        // cubic (indices 6..<8)
        0.0, 0.0,
    ]

    // Golden input: first 3 points from span_points[0] in golden_output/keypoints.json
    private let goldenXY: [[Double]] = [
        [-0.4196018576622009, -0.8514548540115356],
        [-0.3583461046218872, -0.8560490608215332],
        [-0.2848392128944397, -0.8621746301651001],
    ]

    // Expected output from Python: project_xy(goldenXY, goldenPvec)
    private let goldenExpected: [[Double]] = [
        [-1.0106050975917298, -1.7984395847571295],
        [-0.9493224804036131, -1.8026603485889126],
        [-0.8757796183437849, -1.8083377750536596],
    ]

    // MARK: - Cubic clamping

    func testAlphaClamped() {
        // alpha=1.0 should be clamped to 0.5; verify output differs from unclamped
        var pvec = goldenPvec
        pvec[DewarpConfig.cubicIdx.lowerBound] = 1.0  // alpha=1.0 → clamped to 0.5
        pvec[DewarpConfig.cubicIdx.lowerBound + 1] = 0.0  // beta=0.0

        var pvecUnclamped = goldenPvec
        pvecUnclamped[DewarpConfig.cubicIdx.lowerBound] = 0.5  // alpha=0.5 (same as clamped)
        pvecUnclamped[DewarpConfig.cubicIdx.lowerBound + 1] = 0.0

        let xy = [[-0.5, 0.0], [0.0, 0.0], [0.5, 0.0]]
        let clamped = projectXYPure(xyCoords: xy, pvec: pvec)
        let expected = projectXYPure(xyCoords: xy, pvec: pvecUnclamped)

        for i in 0..<xy.count {
            XCTAssertEqual(clamped[i][0], expected[i][0], accuracy: 1e-4,
                           "alpha=1.0 should produce same result as alpha=0.5 (clamped)")
            XCTAssertEqual(clamped[i][1], expected[i][1], accuracy: 1e-4)
        }
    }

    func testBetaClamped() {
        // beta=-2.0 should be clamped to -0.5
        var pvec = goldenPvec
        pvec[DewarpConfig.cubicIdx.lowerBound] = 0.0
        pvec[DewarpConfig.cubicIdx.lowerBound + 1] = -2.0  // clamped to -0.5

        var pvecExpected = goldenPvec
        pvecExpected[DewarpConfig.cubicIdx.lowerBound] = 0.0
        pvecExpected[DewarpConfig.cubicIdx.lowerBound + 1] = -0.5

        let xy = [[0.0, 0.0], [0.3, 0.1]]
        let clamped = projectXYPure(xyCoords: xy, pvec: pvec)
        let expected = projectXYPure(xyCoords: xy, pvec: pvecExpected)

        for i in 0..<xy.count {
            XCTAssertEqual(clamped[i][0], expected[i][0], accuracy: 1e-4)
            XCTAssertEqual(clamped[i][1], expected[i][1], accuracy: 1e-4)
        }
    }

    // MARK: - Horner's method (flat cubic with zero alpha/beta)

    func testZeroCubicProducesZeroZ() {
        // With alpha=0 and beta=0, poly=[0,0,0,0], so z=0 for all x.
        // Points lie on z=0 plane — projection should be consistent with pure pinhole.
        var pvec = goldenPvec
        pvec[DewarpConfig.cubicIdx.lowerBound] = 0.0
        pvec[DewarpConfig.cubicIdx.lowerBound + 1] = 0.0

        let xy = [[0.0, 0.0]]
        let result = projectXYPure(xyCoords: xy, pvec: pvec)
        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0].count, 2)
        // Must return finite values
        XCTAssertFalse(result[0][0].isNaN)
        XCTAssertFalse(result[0][1].isNaN)
    }

    // MARK: - Output shape

    func testOutputCount() {
        let xy = [[0.0, 0.0], [0.5, 0.5], [-0.5, -0.5], [0.3, -0.2]]
        let result = projectXYPure(xyCoords: xy, pvec: goldenPvec)
        XCTAssertEqual(result.count, xy.count, "Output should have same number of points as input")
        for pt in result {
            XCTAssertEqual(pt.count, 2, "Each projected point should be 2D")
        }
    }

    func testSinglePoint() {
        let result = projectXYPure(xyCoords: [[0.0, 0.0]], pvec: goldenPvec)
        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0].count, 2)
    }

    // MARK: - Golden data comparison

    func testGoldenOutput() {
        // Compares projectXYPure output against Python reference values (tolerance 1e-4).
        let result = projectXYPure(xyCoords: goldenXY, pvec: goldenPvec)
        XCTAssertEqual(result.count, goldenExpected.count)
        for i in 0..<goldenExpected.count {
            XCTAssertEqual(result[i][0], goldenExpected[i][0], accuracy: 1e-4,
                           "Point \(i) x mismatch")
            XCTAssertEqual(result[i][1], goldenExpected[i][1], accuracy: 1e-4,
                           "Point \(i) y mismatch")
        }
    }
}
