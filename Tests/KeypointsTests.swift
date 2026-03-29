// KeypointsTests.swift
// Tests for makeKeypointIndex() and projectKeypoints() — ported from keypoints.py

import XCTest
@testable import PageDewarp

final class KeypointsTests: XCTestCase {

    // MARK: - makeKeypointIndex

    /// Verify index structure for a simple span_counts=[3, 4, 2].
    /// nspans=3, npts=9 → index shape (10, 2)
    /// column 1: [0,8,8,8,9,9,9,9,10,10]
    /// column 0: [0,11,12,13,14,15,16,17,18,19]
    func testMakeKeypointIndexSmall() {
        let index = makeKeypointIndex(spanCounts: [3, 4, 2])
        XCTAssertEqual(index.count, 10)  // npts+1

        // Row 0 is origin
        XCTAssertEqual(index[0], [0, 0])

        // column 1: span membership
        let expectedSpan = [0, 8, 8, 8, 9, 9, 9, 9, 10, 10]
        for (i, row) in index.enumerated() {
            XCTAssertEqual(row[1], expectedSpan[i], "row \(i) span index mismatch")
        }

        // column 0: pvec index = 8 + nspans + (row-1)
        // nspans=3, so offset = 8+3 = 11
        for i in 1..<index.count {
            XCTAssertEqual(index[i][0], 11 + (i - 1), "row \(i) pvec index mismatch")
        }
    }

    /// Verify against golden initial_params.json span_counts.
    /// nspans=38, npts=554, so index has 555 rows.
    /// Expected: index[1] = [46, 8], index[14] = [59, 8], index[15] = [60, 9]
    func testMakeKeypointIndexGolden() throws {
        let bundle = Bundle(for: type(of: self))
        guard let url = bundle.url(forResource: "initial_params", withExtension: "json") else {
            throw XCTSkip("initial_params.json not found in bundle")
        }
        let data = try Data(contentsOf: url)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let spanCounts = json["span_counts"] as! [Int]

        let index = makeKeypointIndex(spanCounts: spanCounts)
        let nspans = spanCounts.count
        let npts = spanCounts.reduce(0, +)

        XCTAssertEqual(index.count, npts + 1)
        XCTAssertEqual(index[0], [0, 0])

        // Spot-check known values (computed from Python):
        // index[1] = [8+nspans+0, 8+0] = [46, 8]
        XCTAssertEqual(index[1][0], 8 + nspans)
        XCTAssertEqual(index[1][1], 8)

        // First span has 14 keypoints, so index[14] is the last in span 0
        XCTAssertEqual(index[14][0], 8 + nspans + 13)  // 46+13=59
        XCTAssertEqual(index[14][1], 8)

        // index[15] is first keypoint in span 1
        XCTAssertEqual(index[15][0], 8 + nspans + 14)  // 60
        XCTAssertEqual(index[15][1], 9)
    }

    // MARK: - projectKeypoints

    /// Verify projectKeypoints produces correct shape output (same count as keypointIndex).
    func testProjectKeypointsShape() throws {
        let bundle = Bundle(for: type(of: self))
        guard let url = bundle.url(forResource: "initial_params", withExtension: "json") else {
            throw XCTSkip("initial_params.json not found in bundle")
        }
        let data = try Data(contentsOf: url)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]

        let spanCounts = json["span_counts"] as! [Int]
        let rvec = json["rvec"] as! [Double]
        let tvec = json["tvec"] as! [Double]
        let cubic = json["cubic"] as! [Double]
        let ycoords = json["ycoords"] as! [Double]
        let xcoords = (json["xcoords"] as! [[Double]])

        // Build pvec: [rvec(3), tvec(3), cubic(2), ycoords(38), xcoords flattened]
        var pvec = rvec + tvec + cubic + ycoords
        for xs in xcoords { pvec += xs }

        let keypointIndex = makeKeypointIndex(spanCounts: spanCounts)
        let projected = projectKeypoints(pvec: pvec, keypointIndex: keypointIndex)

        // Output should have same number of rows as keypointIndex
        XCTAssertEqual(projected.count, keypointIndex.count)
        // Each projected point is a 2-element [x, y]
        for pt in projected {
            XCTAssertEqual(pt.count, 2)
        }
    }
}
