// ContourDetectorTests.swift
// Tests for ContourDetector.swift — ported from contours.py

import XCTest
@testable import PageDewarp

final class ContourDetectorTests: XCTestCase {

    // MARK: - intervalMeasureOverlap

    func testOverlapFullyOverlapping() {
        // (0, 10) and (2, 8) → overlap = 6
        let result = intervalMeasureOverlap(intA: (0.0, 10.0), intB: (2.0, 8.0))
        XCTAssertEqual(result, 6.0, accuracy: 1e-10)
    }

    func testOverlapPartiallyOverlapping() {
        // (0, 5) and (3, 8) → overlap = 2
        let result = intervalMeasureOverlap(intA: (0.0, 5.0), intB: (3.0, 8.0))
        XCTAssertEqual(result, 2.0, accuracy: 1e-10)
    }

    func testOverlapNoOverlap() {
        // (0, 3) and (5, 10) → overlap = -2 (negative)
        let result = intervalMeasureOverlap(intA: (0.0, 3.0), intB: (5.0, 10.0))
        XCTAssertEqual(result, -2.0, accuracy: 1e-10)
    }

    func testOverlapTouching() {
        // (0, 5) and (5, 10) → overlap = 0 (just touching)
        let result = intervalMeasureOverlap(intA: (0.0, 5.0), intB: (5.0, 10.0))
        XCTAssertEqual(result, 0.0, accuracy: 1e-10)
    }

    func testOverlapSymmetric() {
        // Should be symmetric: swap intA and intB
        let a = intervalMeasureOverlap(intA: (1.0, 7.0), intB: (4.0, 9.0))
        let b = intervalMeasureOverlap(intA: (4.0, 9.0), intB: (1.0, 7.0))
        XCTAssertEqual(a, b, accuracy: 1e-10)
    }

    // MARK: - blobMeanAndTangent

    func testBlobMeanAndTangentZeroAreaReturnsNil() {
        // A single-point contour has zero area — should return nil.
        let singlePoint = [NSValue(cgPoint: CGPoint(x: 5, y: 5))]
        let result = blobMeanAndTangent(contour: singlePoint)
        XCTAssertNil(result, "Zero-area contour should return nil")
    }

    func testBlobMeanAndTangentHorizontalLineReturnsTangentAlongX() {
        // A wide flat rectangle (much wider than tall) — tangent should point along x-axis.
        // Line contours have zero area; use a closed rectangle instead.
        let pts: [NSValue] = [
            NSValue(cgPoint: CGPoint(x: 0,   y: 49)),
            NSValue(cgPoint: CGPoint(x: 100, y: 49)),
            NSValue(cgPoint: CGPoint(x: 100, y: 51)),
            NSValue(cgPoint: CGPoint(x: 0,   y: 51)),
        ]
        guard let (center, tangent) = blobMeanAndTangent(contour: pts) else {
            return XCTFail("Expected non-nil result for wide flat rectangle")
        }
        // Center should be near midpoint of rectangle
        XCTAssertEqual(center[0], 50.0, accuracy: 2.0)
        XCTAssertEqual(center[1], 50.0, accuracy: 2.0)
        // Tangent should be predominantly along x-axis (|tx| > |ty|)
        XCTAssertGreaterThan(abs(tangent[0]), abs(tangent[1]),
                             "Tangent x-component should dominate for wide flat rectangle")
    }

    func testBlobMeanAndTangentReturns2DValues() {
        // Basic smoke test with a wide flat rectangle — result has 2D center and tangent.
        // Line contours have zero area; use a closed rectangle with non-zero area.
        let pts: [NSValue] = [
            NSValue(cgPoint: CGPoint(x: 0,  y: 9)),
            NSValue(cgPoint: CGPoint(x: 30, y: 9)),
            NSValue(cgPoint: CGPoint(x: 30, y: 11)),
            NSValue(cgPoint: CGPoint(x: 0,  y: 11)),
        ]
        guard let (center, tangent) = blobMeanAndTangent(contour: pts) else {
            return XCTFail("Expected non-nil for valid contour")
        }
        XCTAssertEqual(center.count, 2)
        XCTAssertEqual(tangent.count, 2)
        // Tangent should be a unit vector (approximately)
        let norm = sqrt(tangent[0] * tangent[0] + tangent[1] * tangent[1])
        XCTAssertEqual(norm, 1.0, accuracy: 1e-4, "Tangent should be unit vector")
    }

    // MARK: - ContourInfo

    func testContourInfoAngle() {
        // With tangent = (1, 0), angle should be 0.
        let pts = [NSValue(cgPoint: CGPoint(x: 0, y: 10)),
                   NSValue(cgPoint: CGPoint(x: 100, y: 10))]
        let info = ContourInfo(
            contour: pts,
            rect: CGRect(x: 0, y: 10, width: 100, height: 1),
            center: [50.0, 10.0],
            tangent: [1.0, 0.0]
        )
        XCTAssertEqual(info.angle, 0.0, accuracy: 1e-10)
    }

    func testContourInfoLocalXRange() {
        // Points at x = 0 and x = 100, center at (50, 0), tangent = (1, 0).
        // Projections: 0-50 = -50, 100-50 = +50 → localXRange = (-50, +50)
        let pts = [NSValue(cgPoint: CGPoint(x: 0, y: 0)),
                   NSValue(cgPoint: CGPoint(x: 100, y: 0))]
        let info = ContourInfo(
            contour: pts,
            rect: CGRect(x: 0, y: 0, width: 100, height: 1),
            center: [50.0, 0.0],
            tangent: [1.0, 0.0]
        )
        XCTAssertEqual(info.localXRange.0, -50.0, accuracy: 1e-10)
        XCTAssertEqual(info.localXRange.1, 50.0, accuracy: 1e-10)
    }

    func testContourInfoPoint0Point1() {
        // center = (50, 0), tangent = (1, 0), localMin = -50, localMax = +50
        // point0 = (50 + 1*(-50), 0) = (0, 0)
        // point1 = (50 + 1*(50), 0) = (100, 0)
        let pts = [NSValue(cgPoint: CGPoint(x: 0, y: 0)),
                   NSValue(cgPoint: CGPoint(x: 100, y: 0))]
        let info = ContourInfo(
            contour: pts,
            rect: CGRect(x: 0, y: 0, width: 100, height: 1),
            center: [50.0, 0.0],
            tangent: [1.0, 0.0]
        )
        XCTAssertEqual(info.point0[0], 0.0, accuracy: 1e-10)
        XCTAssertEqual(info.point0[1], 0.0, accuracy: 1e-10)
        XCTAssertEqual(info.point1[0], 100.0, accuracy: 1e-10)
        XCTAssertEqual(info.point1[1], 0.0, accuracy: 1e-10)
    }

    func testContourInfoProjX() {
        // center = (0, 0), tangent = (1, 0): projX([5, 3]) = 5
        let pts = [NSValue(cgPoint: CGPoint(x: -10, y: 0)),
                   NSValue(cgPoint: CGPoint(x: 10, y: 0))]
        let info = ContourInfo(
            contour: pts,
            rect: CGRect(x: -10, y: 0, width: 20, height: 1),
            center: [0.0, 0.0],
            tangent: [1.0, 0.0]
        )
        XCTAssertEqual(info.projX(point: [5.0, 3.0]), 5.0, accuracy: 1e-10)
        XCTAssertEqual(info.projX(point: [-3.0, 7.0]), -3.0, accuracy: 1e-10)
    }

    func testContourInfoLocalOverlap() {
        // Two horizontal contours placed side by side with partial overlap.
        // contour A: center=(0,0), tangent=(1,0), span [-10, 10]
        // contour B: center=(15,0), tangent=(1,0), span [5, 25]
        // B's point0 projected onto A's axis: 5-0 = 5; B's point1 projected: 25-0 = 25
        // overlap of A's (-10,10) with (5,25) = min(10,25) - max(-10,5) = 10-5 = 5
        let ptsA = [NSValue(cgPoint: CGPoint(x: -10, y: 0)),
                    NSValue(cgPoint: CGPoint(x: 10, y: 0))]
        let infoA = ContourInfo(
            contour: ptsA,
            rect: CGRect(x: -10, y: 0, width: 20, height: 1),
            center: [0.0, 0.0],
            tangent: [1.0, 0.0]
        )
        let ptsB = [NSValue(cgPoint: CGPoint(x: 5, y: 0)),
                    NSValue(cgPoint: CGPoint(x: 25, y: 0))]
        let infoB = ContourInfo(
            contour: ptsB,
            rect: CGRect(x: 5, y: 0, width: 20, height: 1),
            center: [15.0, 0.0],
            tangent: [1.0, 0.0]
        )
        // B's point0 = [5,0], point1 = [25,0] (center+tangent*min, center+tangent*max)
        // Projected onto A: projX([5,0]) = 5, projX([25,0]) = 25
        // Overlap of (-10,10) with (5,25) = 10-5 = 5
        let overlap = infoA.localOverlap(other: infoB)
        XCTAssertEqual(overlap, 5.0, accuracy: 1e-10)
    }

    // MARK: - getContours (synthetic mask)

    func testGetContoursOnSyntheticMask() {
        // Create a small grayscale mask with one wide horizontal white rectangle.
        // Rectangle: width=100px, height=3px — should pass all filters.
        let width = 200
        let height = 50
        var pixels = [UInt8](repeating: 0, count: width * height)

        // Draw a 100x3 white rectangle at (50, 20)
        for row in 20..<23 {
            for col in 50..<150 {
                pixels[row * width + col] = 255
            }
        }

        let colorSpace = CGColorSpaceCreateDeviceGray()
        let data = Data(pixels)
        let cfData = data as CFData
        guard let provider = CGDataProvider(data:cfData),
              let cgImage = CGImage(
                  width: width, height: height,
                  bitsPerComponent: 8, bitsPerPixel: 8,
                  bytesPerRow: width, space: colorSpace,
                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                  provider: provider, decode: nil,
                  shouldInterpolate: false, intent: .defaultIntent) else {
            return XCTFail("Failed to create CGImage for synthetic mask")
        }
        let maskImage = UIImage(cgImage: cgImage)

        let contours = getContours(maskImage: maskImage)
        // The 100x3 rectangle should produce at least one contour
        XCTAssertGreaterThan(contours.count, 0, "Should detect at least one contour in mask")

        if let first = contours.first {
            XCTAssertEqual(first.center.count, 2)
            XCTAssertEqual(first.tangent.count, 2)
            // Tangent should be predominantly horizontal
            XCTAssertGreaterThan(abs(first.tangent[0]), abs(first.tangent[1]),
                                 "Tangent should be horizontal for horizontal rectangle")
        }
    }

    func testGetContoursEmptyMaskReturnsEmpty() {
        // All-black mask: no contours.
        let width = 100, height = 50
        let pixels = [UInt8](repeating: 0, count: width * height)
        let colorSpace = CGColorSpaceCreateDeviceGray()
        guard let provider = CGDataProvider(data:Data(pixels) as CFData),
              let cgImage = CGImage(
                  width: width, height: height,
                  bitsPerComponent: 8, bitsPerPixel: 8,
                  bytesPerRow: width, space: colorSpace,
                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                  provider: provider, decode: nil,
                  shouldInterpolate: false, intent: .defaultIntent) else {
            return XCTFail("Failed to create CGImage")
        }
        let maskImage = UIImage(cgImage: cgImage)
        let contours = getContours(maskImage: maskImage)
        XCTAssertEqual(contours.count, 0, "Empty mask should produce no contours")
    }

    func testGetContoursFiltersTooNarrowContours() {
        // A thin 5x3 rectangle — too narrow (textMinWidth=15), should be filtered out.
        let width = 100, height = 50
        var pixels = [UInt8](repeating: 0, count: width * height)
        for row in 20..<23 {
            for col in 10..<15 {  // width=5 < textMinWidth=15
                pixels[row * width + col] = 255
            }
        }
        let colorSpace = CGColorSpaceCreateDeviceGray()
        guard let provider = CGDataProvider(data:Data(pixels) as CFData),
              let cgImage = CGImage(
                  width: width, height: height,
                  bitsPerComponent: 8, bitsPerPixel: 8,
                  bytesPerRow: width, space: colorSpace,
                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                  provider: provider, decode: nil,
                  shouldInterpolate: false, intent: .defaultIntent) else {
            return XCTFail("Failed to create CGImage")
        }
        let maskImage = UIImage(cgImage: cgImage)
        let contours = getContours(maskImage: maskImage)
        XCTAssertEqual(contours.count, 0, "Too-narrow contour should be filtered out")
    }
}
