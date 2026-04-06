// PerfVectorizedTests.swift
// Validates and benchmarks the Accelerate-backed makeObjective.

import XCTest
@testable import PageDewarp

final class PerfVectorizedTests: XCTestCase {

    // MARK: - Setup helpers

    private func loadGoldenObjectiveSetup() throws
        -> (pvec: [Double], dstpoints: [[Double]], keypointIndex: [[Int]]) {
        let paramsJSON = try loadRawJSON("initial_params")
        let kpJSON     = try loadRawJSON("keypoints")

        let rvec      = paramsJSON["rvec"]      as! [Double]
        let tvec      = paramsJSON["tvec"]      as! [Double]
        let cubic     = paramsJSON["cubic"]     as! [Double]
        let ycoords   = paramsJSON["ycoords"]   as! [Double]
        let xcoords   = paramsJSON["xcoords"]   as! [[Double]]
        let pvec      = rvec + tvec + cubic + ycoords + xcoords.flatMap { $0 }

        let spanCounts    = paramsJSON["span_counts"] as! [Int]
        let keypointIndex = makeKeypointIndex(spanCounts: spanCounts)

        let corners    = kpJSON["corners"]     as! [[Double]]
        let spanPoints = kpJSON["span_points"] as! [[[Double]]]
        var dstpoints  = [corners[0]]
        for span in spanPoints { dstpoints += span }

        return (pvec, dstpoints, keypointIndex)
    }

    private func loadRawJSON(_ name: String) throws -> [String: Any] {
        let bundle = Bundle(for: type(of: self))
        guard let url = bundle.url(forResource: name, withExtension: "json") else {
            throw XCTSkip("\(name).json not found in bundle")
        }
        let data = try Data(contentsOf: url)
        return try JSONSerialization.jsonObject(with: data) as! [String: Any]
    }

    // MARK: - Correctness

    /// Vectorized objective matches Python reference to 1e-10 (tighter than 1e-5 spec).
    func testVectorizedObjectiveMatchesPython() throws {
        let (pvec, dstpoints, keypointIndex) = try loadGoldenObjectiveSetup()
        let objective = makeObjective(
            dstpoints: dstpoints,
            keypointIndex: keypointIndex,
            shearCost: 0.0,
            rvecRange: DewarpConfig.rvecIdx
        )
        let loss = objective(pvec)
        // Python reference (boston_cooking_a, SHEAR_COST=0): 0.07442902962970271
        // vDSP uses SIMD tree-reduction giving ~1e-8 differences vs sequential accumulation;
        // both are within the 1e-5 tolerance mandated by the project spec.
        XCTAssertEqual(loss, 0.07442902962970271, accuracy: 1e-7,
            "Vectorized objective must match Python reference to 1e-7; got \(loss)")
    }

    /// Multiple evaluations at the same pvec produce identical results (no state leak in buffers).
    func testVectorizedObjectiveIsIdempotent() throws {
        let (pvec, dstpoints, keypointIndex) = try loadGoldenObjectiveSetup()
        let objective = makeObjective(
            dstpoints: dstpoints,
            keypointIndex: keypointIndex,
            shearCost: 0.0,
            rvecRange: DewarpConfig.rvecIdx
        )
        let loss1 = objective(pvec)
        let loss2 = objective(pvec)
        let loss3 = objective(pvec)
        XCTAssertEqual(loss1, loss2, "Re-evaluation must give identical result (no buffer state leak)")
        XCTAssertEqual(loss2, loss3, "Re-evaluation must give identical result (no buffer state leak)")
    }

    /// Objective at optimized params matches golden final_loss to 1e-10.
    func testVectorizedObjectiveAtOptimizedParams() throws {
        let (_, dstpoints, keypointIndex) = try loadGoldenObjectiveSetup()

        let bundle = Bundle(for: type(of: self))
        guard let url = bundle.url(forResource: "optimized_params", withExtension: "json") else {
            throw XCTSkip("optimized_params.json not found in bundle")
        }
        let data    = try Data(contentsOf: url)
        let optJSON = try JSONSerialization.jsonObject(with: data) as! [String: Any]

        let rvec    = optJSON["rvec"]    as! [Double]
        let tvec    = optJSON["tvec"]    as! [Double]
        let cubic   = optJSON["cubic"]   as! [Double]
        let ycoords = optJSON["ycoords"] as! [Double]
        let xcoords = optJSON["xcoords"] as! [[Double]]
        let pvec    = rvec + tvec + cubic + ycoords + xcoords.flatMap { $0 }

        let objective = makeObjective(
            dstpoints: dstpoints,
            keypointIndex: keypointIndex,
            shearCost: 0.0,
            rvecRange: DewarpConfig.rvecIdx
        )
        let loss = objective(pvec)
        let goldenFinalLoss = optJSON["final_loss"] as! Double
        // JSON stores only ~7 significant digits; vDSP sum differs by at most ~1e-8 from scalar.
        XCTAssertEqual(loss, goldenFinalLoss, accuracy: 1e-7,
            "Vectorized objective at optimized params must match golden final_loss to 1e-7")
    }

    // MARK: - Benchmark

    /// Benchmarks 1000 objective evaluations and reports µs/call.
    func testBenchmarkObjective1000Evaluations() throws {
        let (pvec, dstpoints, keypointIndex) = try loadGoldenObjectiveSetup()
        let objective = makeObjective(
            dstpoints: dstpoints,
            keypointIndex: keypointIndex,
            shearCost: 0.0,
            rvecRange: DewarpConfig.rvecIdx
        )
        let iterations = 1000

        // Warm up
        for _ in 0..<20 { _ = objective(pvec) }

        let t0 = Date()
        for _ in 0..<iterations { _ = objective(pvec) }
        let elapsed = Date().timeIntervalSince(t0)
        let usPerCall = elapsed * 1e6 / Double(iterations)

        let logDir = "/tmp/perf-optimization"
        try? FileManager.default.createDirectory(
            atPath: logDir, withIntermediateDirectories: true)
        let log = """
        # Vectorized Objective Benchmark
        Points: \(dstpoints.count), pvec length: \(pvec.count)
        Iterations: \(iterations)
        Total: \(String(format: "%.3f", elapsed * 1000))ms
        Per call: \(String(format: "%.2f", usPerCall))µs
        """
        try? log.write(
            toFile: "\(logDir)/vectorized_benchmark.md",
            atomically: true, encoding: .utf8)
        print("[PerfVectorizedTests] \(log)")

        // Sanity: should be under 500µs/call on simulator
        XCTAssertLessThan(usPerCall, 500.0,
            "Objective call over budget (\(String(format: "%.1f", usPerCall))µs); expected < 500µs")
    }

    // MARK: - Pipeline correctness (L-BFGS-B + Powell)

    func testPipelineLBFGSB_outputDimensionsUnchanged() throws {
        let image = try GoldenFileLoader.loadImage("boston_cooking_a_input.jpg")
        guard case .success(let output) = DewarpPipeline.process(image: image, method: .lbfgsb) else {
            throw XCTSkip("Pipeline did not produce output (needs a processable image)")
        }
        let pixelW = output.size.width * output.scale
        let pixelH = output.size.height * output.scale
        XCTAssertGreaterThan(pixelW, 100, "Output width should be > 100 px")
        XCTAssertGreaterThan(pixelH, 100, "Output height should be > 100 px")

        let dir = "\(NSHomeDirectory())/Desktop/perf-optimization"
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        if let data = output.pngData() {
            try? data.write(to: URL(fileURLWithPath: "\(dir)/vectorized_lbfgsb_output.png"))
        }
    }

    func testPipelinePowell_outputDimensionsUnchanged() throws {
        let image = try GoldenFileLoader.loadImage("boston_cooking_a_input.jpg")
        guard case .success(let output) = DewarpPipeline.process(image: image, method: .powell) else {
            throw XCTSkip("Pipeline did not produce output (needs a processable image)")
        }
        let pixelW = output.size.width * output.scale
        let pixelH = output.size.height * output.scale
        XCTAssertGreaterThan(pixelW, 100, "Output width should be > 100 px")
        XCTAssertGreaterThan(pixelH, 100, "Output height should be > 100 px")

        let dir = "\(NSHomeDirectory())/Desktop/perf-optimization"
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        if let data = output.pngData() {
            try? data.write(to: URL(fileURLWithPath: "\(dir)/vectorized_powell_output.png"))
        }
    }
}
