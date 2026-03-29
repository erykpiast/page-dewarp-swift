// CameraMatrixTests.swift
// Tests for cameraMatrix() — ported from k_opt.py

import XCTest
@testable import PageDewarp

final class CameraMatrixTests: XCTestCase {

    func testShape() {
        let K = cameraMatrix()
        XCTAssertEqual(K.count, 3)
        XCTAssertEqual(K[0].count, 3)
        XCTAssertEqual(K[1].count, 3)
        XCTAssertEqual(K[2].count, 3)
    }

    func testFocalLengthOnDiagonal() {
        let K = cameraMatrix()
        let f = DewarpConfig.focalLength  // 1.2
        XCTAssertEqual(K[0][0], f, accuracy: 1e-15)
        XCTAssertEqual(K[1][1], f, accuracy: 1e-15)
        XCTAssertEqual(K[2][2], 1.0, accuracy: 1e-15)
    }

    func testOffDiagonalZero() {
        let K = cameraMatrix()
        XCTAssertEqual(K[0][1], 0.0)
        XCTAssertEqual(K[0][2], 0.0)
        XCTAssertEqual(K[1][0], 0.0)
        XCTAssertEqual(K[1][2], 0.0)
        XCTAssertEqual(K[2][0], 0.0)
        XCTAssertEqual(K[2][1], 0.0)
    }

    func testMatchesPythonDefault() {
        // Python: K = [[1.2, 0, 0], [0, 1.2, 0], [0, 0, 1]]
        let K = cameraMatrix()
        XCTAssertEqual(K[0][0], 1.2, accuracy: 1e-15)
        XCTAssertEqual(K[1][1], 1.2, accuracy: 1e-15)
    }
}
