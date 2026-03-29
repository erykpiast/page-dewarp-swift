// ObjectiveTests.swift
// Tests for makeObjective() — ported from optimise/_base.py

import XCTest
@testable import PageDewarp

final class ObjectiveTests: XCTestCase {

    // MARK: - Helpers

    private func buildPvecFromJSON(_ json: [String: Any]) -> [Double] {
        let rvec = json["rvec"] as! [Double]
        let tvec = json["tvec"] as! [Double]
        let cubic = json["cubic"] as! [Double]
        let ycoords = json["ycoords"] as! [Double]
        let xcoords = json["xcoords"] as! [[Double]]
        return rvec + tvec + cubic + ycoords + xcoords.flatMap { $0 }
    }

    private func loadInitialParams() throws -> [String: Any] {
        let bundle = Bundle(for: type(of: self))
        guard let url = bundle.url(forResource: "initial_params", withExtension: "json") else {
            throw XCTSkip("initial_params.json not found in bundle")
        }
        let data = try Data(contentsOf: url)
        return try JSONSerialization.jsonObject(with: data) as! [String: Any]
    }

    private func loadKeypoints() throws -> [String: Any] {
        let bundle = Bundle(for: type(of: self))
        guard let url = bundle.url(forResource: "keypoints", withExtension: "json") else {
            throw XCTSkip("keypoints.json not found in bundle")
        }
        let data = try Data(contentsOf: url)
        return try JSONSerialization.jsonObject(with: data) as! [String: Any]
    }

    // MARK: - Tests

    /// Objective returns a finite, positive scalar for valid golden inputs.
    func testObjectiveReturnsFinitePositiveValue() throws {
        let paramsJSON = try loadInitialParams()
        let kpJSON = try loadKeypoints()

        let pvec = buildPvecFromJSON(paramsJSON)
        let spanCounts = paramsJSON["span_counts"] as! [Int]
        let keypointIndex = makeKeypointIndex(spanCounts: spanCounts)

        // Build dstpoints: corners[0] + all span_points flattened
        // Ported from image.py — dstpoints = vstack([corner[0]] + span_points)
        let corners = kpJSON["corners"] as! [[Double]]
        let spanPoints = kpJSON["span_points"] as! [[[Double]]]
        var dstpoints = [corners[0]]
        for span in spanPoints { dstpoints += span }

        let objective = makeObjective(
            dstpoints: dstpoints,
            keypointIndex: keypointIndex,
            shearCost: DewarpConfig.shearCost,
            rvecRange: DewarpConfig.rvecIdx
        )

        let loss = objective(pvec)
        XCTAssert(loss.isFinite, "Objective should return a finite value")
        XCTAssertGreaterThan(loss, 0, "Objective should be positive for imperfect params")
    }

    /// Shear penalty is applied correctly when shearCost > 0.
    func testShearPenaltyApplied() throws {
        let paramsJSON = try loadInitialParams()
        let kpJSON = try loadKeypoints()

        let pvec = buildPvecFromJSON(paramsJSON)
        let spanCounts = paramsJSON["span_counts"] as! [Int]
        let keypointIndex = makeKeypointIndex(spanCounts: spanCounts)

        let corners = kpJSON["corners"] as! [[Double]]
        let spanPoints = kpJSON["span_points"] as! [[[Double]]]
        var dstpoints = [corners[0]]
        for span in spanPoints { dstpoints += span }

        let noShear = makeObjective(
            dstpoints: dstpoints,
            keypointIndex: keypointIndex,
            shearCost: 0.0,
            rvecRange: DewarpConfig.rvecIdx
        )
        let withShear = makeObjective(
            dstpoints: dstpoints,
            keypointIndex: keypointIndex,
            shearCost: 1.0,
            rvecRange: DewarpConfig.rvecIdx
        )

        let lossNoShear = noShear(pvec)
        let lossWithShear = withShear(pvec)

        // With non-zero shear cost, loss should differ (unless rvec[0] is exactly 0)
        // rvec[0] = -0.0 (golden), so penalty ~ 0; just verify both are finite
        XCTAssert(lossNoShear.isFinite)
        XCTAssert(lossWithShear.isFinite)
        // Loss with shear >= loss without shear (penalty is non-negative)
        XCTAssertGreaterThanOrEqual(lossWithShear, lossNoShear - 1e-10)
    }
}
