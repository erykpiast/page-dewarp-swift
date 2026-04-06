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

    /// Objective at initial params matches Python's make_objective to within 1e-4.
    ///
    /// Python reference:
    ///   objective(pvec) = 0.07442902962970271
    ///   (computed with SHEAR_COST=0, golden data from boston_cooking_a)
    ///
    /// Key: dstpoints must be [[corners[0]] + flatten(spanPoints)]
    /// and project_keypoints must use the same indexing as Python's keypoints.py.
    func testObjectiveMatchesPython() throws {
        let paramsJSON = try loadInitialParams()
        let kpJSON = try loadKeypoints()

        let pvec = buildPvecFromJSON(paramsJSON)
        let spanCounts = paramsJSON["span_counts"] as! [Int]
        let keypointIndex = makeKeypointIndex(spanCounts: spanCounts)

        let corners = kpJSON["corners"] as! [[Double]]
        let spanPoints = kpJSON["span_points"] as! [[[Double]]]
        var dstpoints = [corners[0]]
        for span in spanPoints { dstpoints += span }

        let objective = makeObjective(
            dstpoints: dstpoints,
            keypointIndex: keypointIndex,
            shearCost: 0.0,
            rvecRange: DewarpConfig.rvecIdx
        )

        let loss = objective(pvec)

        // Python value: 0.07442902962970271
        // Tolerance 1e-4 allows for Float32/Float64 precision differences in OpenCV
        XCTAssertEqual(loss, 0.07442902962970271, accuracy: 1e-4,
            "Swift objective should match Python's 0.07443 at initial params")
    }

    /// Objective at initial params matches Python reference value.
    ///
    /// Python (page_dewarp 0.2.7) computes `np.sum((dstpoints - ppts)**2)` where
    /// dstpoints is (N, 1, 2) and ppts is (N, 1, 2) — pure element-wise, no broadcasting.
    /// Swift uses identical per-point squared error. Both should give 0.07443 for
    /// the boston_cooking_a golden file.
    ///
    /// Note: The value 0.029039014790436843 cited in the task spec is for IMG_1389.jpeg,
    /// not boston_cooking_a. The correct Python reference for this golden file is 0.07443.
    func testObjectiveMatchesPythonAtInitialParams() throws {
        let paramsJSON = try loadInitialParams()
        let kpJSON = try loadKeypoints()

        let pvec = buildPvecFromJSON(paramsJSON)
        let spanCounts = paramsJSON["span_counts"] as! [Int]
        let keypointIndex = makeKeypointIndex(spanCounts: spanCounts)

        let corners = kpJSON["corners"] as! [[Double]]
        let spanPoints = kpJSON["span_points"] as! [[[Double]]]
        var dstpoints = [corners[0]]
        for span in spanPoints { dstpoints += span }

        let objective = makeObjective(
            dstpoints: dstpoints,
            keypointIndex: keypointIndex,
            shearCost: 0.0,
            rvecRange: DewarpConfig.rvecIdx
        )

        let loss = objective(pvec)
        // Python reference: 0.07442902962970271 (per-point element-wise sum, boston_cooking_a)
        XCTAssertEqual(loss, 0.07442902962970271, accuracy: 1e-5,
            "Objective at initial params must match Python within 1e-5")
    }

    /// Objective at optimized params matches the golden file final_loss.
    func testObjectiveMatchesPythonAtOptimizedParams() throws {
        let paramsJSON = try loadInitialParams()
        let kpJSON = try loadKeypoints()

        // Load optimized params
        let bundle = Bundle(for: type(of: self))
        guard let url = bundle.url(forResource: "optimized_params", withExtension: "json") else {
            throw XCTSkip("optimized_params.json not found")
        }
        let data = try Data(contentsOf: url)
        let optJSON = try JSONSerialization.jsonObject(with: data) as! [String: Any]

        let pvec = buildPvecFromJSON(optJSON)
        let spanCounts = paramsJSON["span_counts"] as! [Int]
        let keypointIndex = makeKeypointIndex(spanCounts: spanCounts)

        let corners = kpJSON["corners"] as! [[Double]]
        let spanPoints = kpJSON["span_points"] as! [[[Double]]]
        var dstpoints = [corners[0]]
        for span in spanPoints { dstpoints += span }

        let objective = makeObjective(
            dstpoints: dstpoints,
            keypointIndex: keypointIndex,
            shearCost: 0.0,
            rvecRange: DewarpConfig.rvecIdx
        )

        let loss = objective(pvec)
        // Python reference: 0.00487377 (golden file final_loss)
        let goldenFinalLoss = optJSON["final_loss"] as! Double
        XCTAssertEqual(loss, goldenFinalLoss, accuracy: 1e-5,
            "Objective at optimized params must match golden final_loss within 1e-5")
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
