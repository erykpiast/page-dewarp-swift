// SpanAssemblerTests.swift
// Tests for SpanAssembler.swift — ported from spans.py

import XCTest
@testable import PageDewarp

final class SpanAssemblerTests: XCTestCase {

    // MARK: - angleDist

    func testAngleDistZero() {
        // Same angle → distance = 0
        XCTAssertEqual(angleDist(0.5, 0.5), 0.0, accuracy: 1e-10)
    }

    func testAngleDistPositive() {
        // 0.3 - 0.1 = 0.2
        XCTAssertEqual(angleDist(0.3, 0.1), 0.2, accuracy: 1e-10)
    }

    func testAngleDistWrapsAcrossPi() {
        // π - (-π + 0.1) wraps: diff = 2π - 0.1 → subtract 2π → -0.1 → abs = 0.1
        let result = angleDist(.pi, -.pi + 0.1)
        XCTAssertEqual(result, 0.1, accuracy: 1e-10)
    }

    func testAngleDistSymmetric() {
        // angleDist is symmetric: |a - b| = |b - a| (mod 2π)
        let ab = angleDist(1.0, 0.3)
        let ba = angleDist(0.3, 1.0)
        XCTAssertEqual(ab, ba, accuracy: 1e-10)
    }

    func testAngleDistNeverNegative() {
        XCTAssertGreaterThanOrEqual(angleDist(-1.0, 2.5), 0.0)
        XCTAssertGreaterThanOrEqual(angleDist(3.0, -3.0), 0.0)
    }

    // MARK: - generateCandidateEdge

    private func makeHorizontalContour(x: Double, y: Double, width: Double) -> ContourInfo {
        let pts = [
            NSValue(cgPoint: CGPoint(x: x, y: y)),
            NSValue(cgPoint: CGPoint(x: x + width, y: y)),
        ]
        return ContourInfo(
            contour: pts,
            rect: CGRect(x: x, y: y - 1, width: width, height: 3),
            center: [x + width / 2, y],
            tangent: [1.0, 0.0]
        )
    }

    func testCandidateEdgeValidPairProducesEdge() {
        // Two horizontal contours 10px apart, same row — should produce an edge.
        let a = makeHorizontalContour(x: 0, y: 50, width: 40)
        let b = makeHorizontalContour(x: 50, y: 50, width: 40)
        let edge = generateCandidateEdge(a, b)
        XCTAssertNotNil(edge, "Close, aligned contours should form a valid edge")
    }

    func testCandidateEdgeRejectsTooFarApart() {
        // Distance > edgeMaxLength (100) → should be rejected.
        let a = makeHorizontalContour(x: 0, y: 50, width: 40)
        let b = makeHorizontalContour(x: 200, y: 50, width: 40)  // dist = 160 > 100
        let edge = generateCandidateEdge(a, b)
        XCTAssertNil(edge, "Contours >100px apart should be rejected")
    }

    func testCandidateEdgeRejectsTooMuchOverlap() {
        // Two overlapping contours (overlap > edgeMaxOverlap=1.0) → rejected.
        // Both span from x=0..80, center at x=40 — heavy overlap.
        let a = makeHorizontalContour(x: 0, y: 50, width: 80)
        let b = makeHorizontalContour(x: 20, y: 50, width: 80)  // overlap = 60
        let edge = generateCandidateEdge(a, b)
        XCTAssertNil(edge, "Heavily overlapping contours should be rejected")
    }

    func testCandidateEdgeScoreIncludesDistance() {
        // Two adjacent contours — score should be non-negative and finite.
        let a = makeHorizontalContour(x: 0, y: 50, width: 40)
        let b = makeHorizontalContour(x: 45, y: 50, width: 40)
        guard let (score, _, _) = generateCandidateEdge(a, b) else {
            return XCTFail("Expected valid edge for adjacent contours")
        }
        XCTAssertGreaterThanOrEqual(score, 0.0)
        XCTAssertFalse(score.isNaN)
    }

    func testCandidateEdgeSwapsOrder() {
        // Passing (b, a) instead of (a, b) should still return valid edge with same score.
        let a = makeHorizontalContour(x: 0, y: 50, width: 40)
        let b = makeHorizontalContour(x: 45, y: 50, width: 40)
        guard let (scoreAB, _, _) = generateCandidateEdge(a, b),
              let (scoreBA, _, _) = generateCandidateEdge(b, a) else {
            return XCTFail("Both orderings should produce valid edges")
        }
        XCTAssertEqual(scoreAB, scoreBA, accuracy: 1e-10)
    }

    // MARK: - assembleSpans

    func testAssembleSpansLinksContiguousContours() {
        // Three sequential horizontal contours → should form one span.
        let a = makeHorizontalContour(x: 0,  y: 50, width: 40)
        let b = makeHorizontalContour(x: 45, y: 50, width: 40)
        let c = makeHorizontalContour(x: 90, y: 50, width: 40)
        let spans = assembleSpans(contours: [a, b, c])
        // Total width = 40+40+40 = 120 > spanMinWidth(30) → one span
        XCTAssertEqual(spans.count, 1, "Three adjacent contours should form one span")
        XCTAssertEqual(spans[0].count, 3)
    }

    func testAssembleSpansTwoSeparateRows() {
        // Two rows of contours far apart vertically → should form two separate spans.
        let a1 = makeHorizontalContour(x: 0,  y: 30,  width: 40)
        let a2 = makeHorizontalContour(x: 45, y: 30,  width: 40)
        let b1 = makeHorizontalContour(x: 0,  y: 200, width: 40)
        let b2 = makeHorizontalContour(x: 45, y: 200, width: 40)
        let spans = assembleSpans(contours: [a1, a2, b1, b2])
        XCTAssertEqual(spans.count, 2, "Two separate rows should produce two spans")
    }

    func testAssembleSpansFiltersNarrowChains() {
        // A very short chain (width < spanMinWidth=30) should be filtered out.
        let a = makeHorizontalContour(x: 0, y: 50, width: 10)  // total width = 10 < 30
        let spans = assembleSpans(contours: [a])
        XCTAssertEqual(spans.count, 0, "Single narrow contour should be filtered out")
    }

    func testAssembleSpansEmptyInput() {
        let spans = assembleSpans(contours: [])
        XCTAssertEqual(spans.count, 0)
    }

    // MARK: - sampleSpans (smoke test with OpenCV bridge)

    func testSampleSpansReturnsNormalizedCoords() {
        // Create a synthetic UIImage mask with a horizontal white rectangle.
        // Then detect contours, assemble spans, and verify sampled points are normalized.
        let width = 300, height = 50
        var pixels = [UInt8](repeating: 0, count: width * height)
        // Draw a 200x4 white rectangle at (50, 20).
        for row in 20..<24 {
            for col in 50..<250 {
                pixels[row * width + col] = 255
            }
        }
        guard let maskImage = makeMaskImage(pixels: pixels, width: width, height: height) else {
            return XCTFail("Failed to create mask image")
        }

        let contours = getContours(maskImage: maskImage)
        guard !contours.isEmpty else {
            return XCTFail("Expected at least one contour in synthetic mask")
        }

        // Assemble into spans.
        let spans = assembleSpans(contours: contours)
        guard !spans.isEmpty else {
            return XCTFail("Expected at least one span")
        }

        // Sample span points.
        let shape = (height: height, width: width)
        let spanPoints = sampleSpans(shape: shape, spans: spans)
        XCTAssertFalse(spanPoints.isEmpty, "sampleSpans should return at least one span")

        // All normalized coordinates should lie roughly in [-1, 1].
        for points in spanPoints {
            for pt in points {
                XCTAssertEqual(pt.count, 2)
                XCTAssertLessThanOrEqual(abs(pt[0]), 1.5, "Normalized x should be near [-1,1]")
                XCTAssertLessThanOrEqual(abs(pt[1]), 1.5, "Normalized y should be near [-1,1]")
            }
        }
    }

    // MARK: - keypointsFromSamples (smoke test)

    func testKeypointsFromSamplesReturnsFourCorners() {
        // Simple normalized span points forming a horizontal line.
        let spanPts: [[[Double]]] = [
            [[-0.5, 0.0], [-0.3, 0.01], [-0.1, 0.0], [0.1, -0.01], [0.3, 0.0], [0.5, 0.01]],
        ]
        // Page outline: a rectangle in pixel space (400x300 image).
        let pageOutline: [NSValue] = [
            NSValue(cgPoint: CGPoint(x: 10,  y: 10)),
            NSValue(cgPoint: CGPoint(x: 390, y: 10)),
            NSValue(cgPoint: CGPoint(x: 390, y: 290)),
            NSValue(cgPoint: CGPoint(x: 10,  y: 290)),
        ]
        let pagemask = (height: 300, width: 400)
        let (corners, ycoords, xcoords) = keypointsFromSamples(
            pagemask: pagemask,
            pageOutline: pageOutline,
            spanPoints: spanPts
        )
        XCTAssertEqual(corners.count, 4, "Should return 4 corners")
        for corner in corners {
            XCTAssertEqual(corner.count, 2)
        }
        XCTAssertEqual(ycoords.count, 1, "Should return one y-coord per span")
        XCTAssertEqual(xcoords.count, 1, "Should return one x-coord array per span")
    }
}

// MARK: - Test helpers

private func makeMaskImage(pixels: [UInt8], width: Int, height: Int) -> UIImage? {
    let colorSpace = CGColorSpaceCreateDeviceGray()
    guard let provider = CGDataProvider(data: Data(pixels) as CFData),
          let cgImage = CGImage(
              width: width, height: height,
              bitsPerComponent: 8, bitsPerPixel: 8,
              bytesPerRow: width, space: colorSpace,
              bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
              provider: provider, decode: nil,
              shouldInterpolate: false, intent: .defaultIntent) else {
        return nil
    }
    return UIImage(cgImage: cgImage)
}
