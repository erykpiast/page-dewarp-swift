import XCTest
@testable import PageDewarp

final class NormalisationTests: XCTestCase {

    let shape = (height: 700, width: 1280)

    // MARK: - pix2norm tests

    func testPix2normCenter() {
        // Center of image should map to (0, 0) in normalized space
        let center = [[Double(shape.width) / 2.0, Double(shape.height) / 2.0]]
        let result = pix2norm(shape: shape, pts: center)
        XCTAssertEqual(result[0][0], 0.0, accuracy: 1e-10)
        XCTAssertEqual(result[0][1], 0.0, accuracy: 1e-10)
    }

    func testPix2normOrigin() {
        // Top-left corner (0, 0)
        let pts = [[0.0, 0.0]]
        let result = pix2norm(shape: shape, pts: pts)
        // scl = 2.0 / max(700, 1280) = 2.0 / 1280
        let scl = 2.0 / Double(max(shape.height, shape.width))
        let expectedX = (0.0 - Double(shape.width) * 0.5) * scl
        let expectedY = (0.0 - Double(shape.height) * 0.5) * scl
        XCTAssertEqual(result[0][0], expectedX, accuracy: 1e-10)
        XCTAssertEqual(result[0][1], expectedY, accuracy: 1e-10)
    }

    func testPix2normBottomRight() {
        // Bottom-right corner (width, height)
        let pts = [[Double(shape.width), Double(shape.height)]]
        let result = pix2norm(shape: shape, pts: pts)
        let scl = 2.0 / Double(max(shape.height, shape.width))
        let expectedX = (Double(shape.width) - Double(shape.width) * 0.5) * scl
        let expectedY = (Double(shape.height) - Double(shape.height) * 0.5) * scl
        XCTAssertEqual(result[0][0], expectedX, accuracy: 1e-10)
        XCTAssertEqual(result[0][1], expectedY, accuracy: 1e-10)
    }

    func testPix2normMultiplePoints() {
        let pts = [[0.0, 0.0], [640.0, 350.0], [1280.0, 700.0]]
        let result = pix2norm(shape: shape, pts: pts)
        XCTAssertEqual(result.count, 3)
        // Center should be (0, 0)
        XCTAssertEqual(result[1][0], 0.0, accuracy: 1e-10)
        XCTAssertEqual(result[1][1], 0.0, accuracy: 1e-10)
    }

    // MARK: - norm2pix tests

    func testNorm2pixOriginMapsToCenter() {
        // Normalized (0, 0) should map to image center
        let pts = [[0.0, 0.0]]
        let result = norm2pix(shape: shape, pts: pts, asInteger: false)
        XCTAssertEqual(result[0][0], Double(shape.width) / 2.0, accuracy: 1e-10)
        XCTAssertEqual(result[0][1], Double(shape.height) / 2.0, accuracy: 1e-10)
    }

    func testNorm2pixAsInteger() {
        let pts = [[0.0, 0.0]]
        let result = norm2pix(shape: shape, pts: pts, asInteger: true)
        // Should be integer-valued
        XCTAssertEqual(result[0][0], 640.0, accuracy: 1e-10)
        XCTAssertEqual(result[0][1], 350.0, accuracy: 1e-10)
    }

    func testNorm2pixNegativePixelCoordinatesTruncateTowardZero() {
        // Use a tiny shape so normalized coords produce negative pixel values.
        // shape (2, 2): scl=1, offsetX=1, offsetY=1
        // norm (-1.6, -1.6) -> pixel (-0.6, -0.6)
        // Python int(-0.6 + 0.5) = int(-0.1) = 0 (truncates toward zero)
        // Swift Int(-0.6 + 0.5)  = Int(-0.1) = 0 (same: truncates toward zero)
        let tinyShape = (height: 2, width: 2)
        let pts = [[-1.6, -1.6]]
        let result = norm2pix(shape: tinyShape, pts: pts, asInteger: true)
        XCTAssertEqual(result[0][0], 0.0, accuracy: 1e-10)
        XCTAssertEqual(result[0][1], 0.0, accuracy: 1e-10)
    }

    // MARK: - Roundtrip tests

    func testRoundtripFloat() {
        let original = [[100.0, 200.0], [640.0, 350.0], [1200.0, 600.0]]
        let normalized = pix2norm(shape: shape, pts: original)
        let recovered = norm2pix(shape: shape, pts: normalized, asInteger: false)
        for (orig, rec) in zip(original, recovered) {
            XCTAssertEqual(orig[0], rec[0], accuracy: 1e-10)
            XCTAssertEqual(orig[1], rec[1], accuracy: 1e-10)
        }
    }

    func testRoundtripInteger() {
        // For integer roundtrip, original points must be integer-valued
        let original = [[100.0, 200.0], [640.0, 350.0], [1200.0, 600.0]]
        let normalized = pix2norm(shape: shape, pts: original)
        let recovered = norm2pix(shape: shape, pts: normalized, asInteger: true)
        for (orig, rec) in zip(original, recovered) {
            XCTAssertEqual(orig[0], rec[0], accuracy: 0.5)
            XCTAssertEqual(orig[1], rec[1], accuracy: 0.5)
        }
    }

    // MARK: - Scale consistency

    func testScaleSymmetry() {
        // Landscape image: longer side is width (1280), scl = 2/1280
        let landscapeShape = (height: 700, width: 1280)
        let pts = [[1280.0, 0.0]]
        let result = pix2norm(shape: landscapeShape, pts: pts)
        // x should be +1.0
        XCTAssertEqual(result[0][0], 1.0, accuracy: 1e-10)
    }

    func testPortraitScale() {
        // Portrait image: longer side is height (1280), scl = 2/1280
        let portraitShape = (height: 1280, width: 700)
        let pts = [[0.0, 1280.0]]
        let result = pix2norm(shape: portraitShape, pts: pts)
        // y should be +1.0
        XCTAssertEqual(result[0][1], 1.0, accuracy: 1e-10)
    }
}
