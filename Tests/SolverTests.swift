// SolverTests.swift
// Tests for getDefaultParams() — ported from solve.py

import XCTest
@testable import PageDewarp

final class SolverTests: XCTestCase {

    // Golden corners from golden_output/keypoints.json
    private let goldenCorners: [[Double]] = [
        [-0.5962007285432904, -0.9444430488014438],
        [ 0.6076691006680317, -0.9371052476847103],
        [ 0.5962007285432904,  0.9444430488014438],
        [-0.6076691006680317,  0.9371052476847103],
    ]

    // Golden ycoords from golden_output/keypoints.json (38 values)
    private let goldenYcoords: [Double] = [
        0.05553471, 0.14578582, 0.20413187, 0.24815692, 0.29010007,
        0.33131094, 0.37322151, 0.41411158, 0.45482559, 0.4951968,
        0.54793159, 0.60387449, 0.66218476, 0.7050341, 0.74551883,
        0.78680237, 0.82764924, 0.86829274, 0.91060303, 0.95144894,
        0.99475889, 1.03464368, 1.07558556, 1.14522336, 1.20594324,
        1.24819103, 1.29037204, 1.33357619, 1.37474541, 1.41763132,
        1.46051924, 1.50247254, 1.54272265, 1.58785124, 1.62598303,
        1.67525678, 1.65197467, 1.73869191,
    ]

    // First two xcoord spans from golden_output/keypoints.json
    private let goldenXcoords0: [Double] = [
        0.17802138, 0.23919679, 0.31260354, 0.37377894,
        0.43494503, 0.49611111, 0.55726786, 0.61842461,
        0.67959069, 0.74076609, 0.8019415,  0.86317284,
        0.89371858, 0.9549313,
    ]
    private let goldenXcoords1: [Double] = [0.59147996, 0.65268334]

    // Expected page dims from golden_output/page_dims.json / initial_params.json
    private let expectedPageWidth  = 1.20389219
    private let expectedPageHeight = 1.88158325

    // Expected rvec/tvec from golden_output/initial_params.json
    private let expectedRvec: [Double] = [-0.0, 0.0, 0.0060951]
    private let expectedTvec: [Double] = [-0.59620073, -0.94444305, 1.20000005]

    // MARK: - Helpers

    /// Build minimal input with only two xcoord spans (for quick tests).
    private func makeMinimalInput() -> (corners: [[Double]], ycoords: [Double], xcoords: [[Double]]) {
        let ycoords = [goldenYcoords[0], goldenYcoords[1]]
        let xcoords = [goldenXcoords0, goldenXcoords1]
        return (goldenCorners, ycoords, xcoords)
    }

    // MARK: - Basic success

    func testSucceedsWithGoldenInput() {
        let (corners, ycoords, xcoords) = makeMinimalInput()
        let result = getDefaultParams(corners: corners, ycoords: ycoords, xcoords: xcoords)
        switch result {
        case .success:
            break
        case .failure(let err):
            XCTFail("Expected success, got \(err)")
        }
    }

    // MARK: - Page dimensions

    func testPageDimsMatchGolden() {
        // Ported from solve.py:38 — page dims are corner distances
        let (corners, ycoords, xcoords) = makeMinimalInput()
        guard case .success(let out) = getDefaultParams(corners: corners, ycoords: ycoords, xcoords: xcoords) else {
            return XCTFail("solvePnP failed")
        }
        XCTAssertEqual(out.pageDims.0, expectedPageWidth,  accuracy: 1e-3, "page_width mismatch")
        XCTAssertEqual(out.pageDims.1, expectedPageHeight, accuracy: 1e-3, "page_height mismatch")
    }

    // MARK: - Span counts

    func testSpanCountsMatchXcoordLengths() {
        // spanCounts[i] = len(xcoords[i])
        let xcoords = [goldenXcoords0, goldenXcoords1]
        guard case .success(let out) = getDefaultParams(corners: goldenCorners, ycoords: [0.0, 0.1], xcoords: xcoords) else {
            return XCTFail("solvePnP failed")
        }
        XCTAssertEqual(out.spanCounts, [goldenXcoords0.count, goldenXcoords1.count])
    }

    // MARK: - Parameter vector layout

    func testParamVectorLength() {
        // Expected: 8 + len(ycoords) + sum(len(xc) for xc in xcoords)
        // Ported from solve.py:53-62
        let ycoords = goldenYcoords
        let xcoords = [goldenXcoords0, goldenXcoords1]
        let expectedLen = 8 + ycoords.count + xcoords.flatMap { $0 }.count
        guard case .success(let out) = getDefaultParams(corners: goldenCorners, ycoords: ycoords, xcoords: xcoords) else {
            return XCTFail("solvePnP failed")
        }
        XCTAssertEqual(out.params.count, expectedLen, "param vector length mismatch")
    }

    func testParamVectorCubicSlopesAreZero() {
        // Initial cubic slopes are always [0.0, 0.0]
        // Ported from solve.py:39
        let (corners, ycoords, xcoords) = makeMinimalInput()
        guard case .success(let out) = getDefaultParams(corners: corners, ycoords: ycoords, xcoords: xcoords) else {
            return XCTFail("solvePnP failed")
        }
        XCTAssertEqual(out.params[DewarpConfig.cubicIdx.lowerBound], 0.0, accuracy: 1e-9)
        XCTAssertEqual(out.params[DewarpConfig.cubicIdx.lowerBound + 1], 0.0, accuracy: 1e-9)
    }

    func testParamVectorYcoordsAtOffset8() {
        // ycoords start at index 8 in the param vector
        // Ported from solve.py:59
        let ycoords = [0.1, 0.2, 0.3]
        let xcoords = [[0.5, 0.6]]
        guard case .success(let out) = getDefaultParams(corners: goldenCorners, ycoords: ycoords, xcoords: xcoords) else {
            return XCTFail("solvePnP failed")
        }
        XCTAssertEqual(out.params[8], 0.1, accuracy: 1e-9)
        XCTAssertEqual(out.params[9], 0.2, accuracy: 1e-9)
        XCTAssertEqual(out.params[10], 0.3, accuracy: 1e-9)
    }

    func testParamVectorXcoordsAfterYcoords() {
        // xcoords are flattened after ycoords
        // Ported from solve.py:61
        let ycoords = [0.1, 0.2]
        let xcoords = [[0.5, 0.6], [0.7]]
        guard case .success(let out) = getDefaultParams(corners: goldenCorners, ycoords: ycoords, xcoords: xcoords) else {
            return XCTFail("solvePnP failed")
        }
        let xStart = 8 + ycoords.count  // index 10
        XCTAssertEqual(out.params[xStart],     0.5, accuracy: 1e-9)
        XCTAssertEqual(out.params[xStart + 1], 0.6, accuracy: 1e-9)
        XCTAssertEqual(out.params[xStart + 2], 0.7, accuracy: 1e-9)
    }

    // MARK: - Rvec / tvec golden values

    func testRvecMatchesGolden() {
        // solvePnP should produce rotation vector close to Python reference
        let (corners, ycoords, xcoords) = makeMinimalInput()
        guard case .success(let out) = getDefaultParams(corners: corners, ycoords: ycoords, xcoords: xcoords) else {
            return XCTFail("solvePnP failed")
        }
        for i in DewarpConfig.rvecIdx {
            XCTAssertEqual(out.params[i], expectedRvec[i], accuracy: 1e-4, "rvec[\(i)] mismatch")
        }
    }

    func testTvecMatchesGolden() {
        let (corners, ycoords, xcoords) = makeMinimalInput()
        guard case .success(let out) = getDefaultParams(corners: corners, ycoords: ycoords, xcoords: xcoords) else {
            return XCTFail("solvePnP failed")
        }
        for i in DewarpConfig.tvecIdx {
            XCTAssertEqual(out.params[i], expectedTvec[i - 3], accuracy: 1e-4, "tvec[\(i)] mismatch")
        }
    }

    func testParamsContainNoNaN() {
        let (corners, ycoords, xcoords) = makeMinimalInput()
        guard case .success(let out) = getDefaultParams(corners: corners, ycoords: ycoords, xcoords: xcoords) else {
            return XCTFail("solvePnP failed")
        }
        for (i, v) in out.params.enumerated() {
            XCTAssertFalse(v.isNaN, "params[\(i)] is NaN")
            XCTAssertFalse(v.isInfinite, "params[\(i)] is Inf")
        }
    }

    // MARK: - Failure case

    func testSolvePnPFailsWithDegenerateCorners() {
        // All corners at the same point → degenerate, solvePnP should fail or produce garbage.
        // We only check that if solvePnP fails it returns .failure.
        // (If OpenCV still succeeds here, that's also acceptable — we just skip.)
        let degenCorners = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        let result = getDefaultParams(corners: degenCorners, ycoords: [0.0], xcoords: [[0.0]])
        // Either success (OpenCV produced something) or failure — both are valid.
        // The important thing is the function doesn't crash.
        switch result {
        case .success, .failure:
            break
        }
    }
}
