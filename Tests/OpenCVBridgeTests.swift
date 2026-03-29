import XCTest
@testable import PageDewarp

final class OpenCVBridgeTests: XCTestCase {

    // MARK: - Version

    func testOpenCVVersionIsNonEmpty() {
        let version = OpenCVWrapper.versionString()
        XCTAssertFalse(version.isEmpty, "OpenCV version string must be non-empty")
        print("Linked OpenCV version: \(version)")
    }

    func testOpenCVVersionFormat() {
        let version = OpenCVWrapper.versionString()
        let components = version.split(separator: ".")
        XCTAssertGreaterThanOrEqual(components.count, 2, "Expected semver format like '4.9.0'")
    }

    // MARK: - SVDecomp

    func testSVDecompOf2x2IdentityMatrix() {
        // SVD of I should give w = [1, 1] and orthogonal u, vt.
        let identity: [NSNumber] = [1, 0, 0, 1]
        let result = OpenCVWrapper.svDecomp(ofMatrix: identity, rows: 2, cols: 2)
        guard let w = result["w"] else { return XCTFail("Missing w") }
        XCTAssertEqual(w.count, 2)
        XCTAssertEqual(w[0].doubleValue, 1.0, accuracy: 1e-6)
        XCTAssertEqual(w[1].doubleValue, 1.0, accuracy: 1e-6)
    }

    // MARK: - PCACompute

    func testPCAComputeReturnsMeanAndEigenvectors() {
        // Horizontal line of points — first eigenvector should be ~(1, 0).
        let points = (0..<10).map { i -> NSValue in
            NSValue(cgPoint: CGPoint(x: Double(i), y: 0.0))
        }
        let result = OpenCVWrapper.pcaCompute(onPoints: points)
        guard let mean = result["mean"], let eigvec = result["eigenvectors"] else {
            return XCTFail("Missing keys")
        }
        XCTAssertEqual(mean.count, 2)
        XCTAssertEqual(eigvec.count, 4)
        // First eigenvector x-component should be dominant.
        let ex = abs(eigvec[0].doubleValue)
        XCTAssertGreaterThan(ex, 0.9, "First eigenvector should point along x-axis")
    }

    // MARK: - ConvexHull

    func testConvexHullOfSquare() {
        let pts = [
            NSValue(cgPoint: CGPoint(x: 0, y: 0)),
            NSValue(cgPoint: CGPoint(x: 1, y: 0)),
            NSValue(cgPoint: CGPoint(x: 1, y: 1)),
            NSValue(cgPoint: CGPoint(x: 0, y: 1)),
            NSValue(cgPoint: CGPoint(x: 0.5, y: 0.5)), // interior point
        ]
        let hull = OpenCVWrapper.convexHull(ofPoints: pts)
        // Interior point should not be in the hull.
        XCTAssertEqual(hull.count, 4, "Square hull should have 4 points")
    }

    // MARK: - Rodrigues

    func testRodriguesZeroVectorIsIdentity() {
        // Rodrigues([0,0,0]) → identity matrix.
        let rvec: [NSNumber] = [0, 0, 0]
        let mat = OpenCVWrapper.rodrigues(fromVector: rvec)
        XCTAssertEqual(mat.count, 9)
        let expected = [1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 1.0]
        for (i, v) in mat.enumerated() {
            XCTAssertEqual(v.doubleValue, expected[i], accuracy: 1e-6,
                           "Rodrigues(0) should be identity at index \(i)")
        }
    }

    // MARK: - ProjectPoints

    func testProjectPointsOriginStaysAtPrincipalPoint() {
        // A single point at world origin with identity rotation/translation.
        // With fx=fy=1, cx=cy=0, it should project to (0,0).
        let pts3D: [NSNumber] = [0, 0, 1]  // z=1 for valid projection
        let rvec: [NSNumber] = [0, 0, 0]
        let tvec: [NSNumber] = [0, 0, 0]
        // Camera matrix: fx=1, fy=1, cx=0, cy=0
        let K: [NSNumber] = [1, 0, 0,
                              0, 1, 0,
                              0, 0, 1]
        let dist: [NSNumber] = [0, 0, 0, 0, 0]
        let projected = OpenCVWrapper.projectPointsWith3DPoints(pts3D, rvec: rvec,
                                                    tvec: tvec, cameraMatrix: K,
                                                    distCoeffs: dist)
        XCTAssertEqual(projected.count, 1)
        let pt = projected[0].cgPointValue
        XCTAssertEqual(pt.x, 0.0, accuracy: 1e-6)
        XCTAssertEqual(pt.y, 0.0, accuracy: 1e-6)
    }

    // MARK: - AdaptiveThreshold

    func testAdaptiveThresholdProducesGrayscaleImage() {
        // Create a simple 32x32 white image and threshold it.
        let size = CGSize(width: 32, height: 32)
        UIGraphicsBeginImageContextWithOptions(size, true, 1.0)
        UIColor.white.setFill()
        UIRectFill(CGRect(origin: .zero, size: size))
        let img = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()

        let result = OpenCVWrapper.adaptiveThresholdImage(img, maxValue: 255, blockSize: 11, c: 2)
        XCTAssertNotNil(result, "adaptiveThreshold should return a non-nil image")
    }

    // MARK: - Resize

    func testResizeImage() {
        let size = CGSize(width: 64, height: 64)
        UIGraphicsBeginImageContextWithOptions(size, true, 1.0)
        UIColor.gray.setFill()
        UIRectFill(CGRect(origin: .zero, size: size))
        let img = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()

        let resized = OpenCVWrapper.resize(img, width: 32, height: 32, interpolation: 1)
        XCTAssertNotNil(resized)
        XCTAssertEqual(Int(resized!.size.width), 32)
        XCTAssertEqual(Int(resized!.size.height), 32)
    }
}
