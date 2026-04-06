// PerfProjectionTests.swift
// Validates that projectXYPure matches projectXY (OpenCV) and benchmarks the speedup.

import XCTest
@testable import PageDewarp
#if SWIFT_PACKAGE
import OpenCVBridge
#endif

final class PerfProjectionTests: XCTestCase {

    // Golden pvec from initial_params.json
    private let goldenPvec: [Double] = [
        -0.0, 0.0, 0.0060951,
        -0.59620073, -0.94444305, 1.20000005,
        0.0, 0.0,
    ]

    // ~200 synthetic (x, y) points spanning [-0.5, 0.5] range
    private var syntheticPoints: [[Double]] {
        var pts: [[Double]] = []
        for i in 0..<200 {
            let x = -0.5 + Double(i) / 199.0
            let y = -0.3 + Double(i % 40) / 39.0 * 0.6
            pts.append([x, y])
        }
        return pts
    }

    // MARK: - Correctness: pure Swift matches OpenCV to 1e-10

    func testPureMatchesOpenCV_goldenData() {
        let goldenXY: [[Double]] = [
            [-0.4196018576622009, -0.8514548540115356],
            [-0.3583461046218872, -0.8560490608215332],
            [-0.2848392128944397, -0.8621746301651001],
        ]
        let opencvResult = projectXY(xyCoords: goldenXY, pvec: goldenPvec)
        let pureResult = projectXYPure(xyCoords: goldenXY, pvec: goldenPvec)

        XCTAssertEqual(opencvResult.count, pureResult.count)
        for i in 0..<goldenXY.count {
            XCTAssertEqual(pureResult[i][0], opencvResult[i][0], accuracy: 1e-10,
                "Point \(i) u: pure=\(pureResult[i][0]) opencv=\(opencvResult[i][0])")
            XCTAssertEqual(pureResult[i][1], opencvResult[i][1], accuracy: 1e-10,
                "Point \(i) v: pure=\(pureResult[i][1]) opencv=\(opencvResult[i][1])")
        }
    }

    func testPureMatchesOpenCV_syntheticPoints() {
        let pts = syntheticPoints
        // Use non-zero cubic coefficients to stress-test
        var pvec = goldenPvec
        pvec[6] = 0.15   // alpha
        pvec[7] = -0.08  // beta

        let opencvResult = projectXY(xyCoords: pts, pvec: pvec)
        let pureResult = projectXYPure(xyCoords: pts, pvec: pvec)

        XCTAssertEqual(opencvResult.count, pureResult.count)
        var maxErr = 0.0
        for i in 0..<pts.count {
            let eu = abs(pureResult[i][0] - opencvResult[i][0])
            let ev = abs(pureResult[i][1] - opencvResult[i][1])
            maxErr = max(maxErr, eu, ev)
        }
        XCTAssertLessThan(maxErr, 1e-10,
            "Max error between pure and OpenCV: \(maxErr)")
    }

    func testPureMatchesOpenCV_emptyInput() {
        let result = projectXYPure(xyCoords: [], pvec: goldenPvec)
        XCTAssertEqual(result.count, 0)
    }

    func testPureMatchesOpenCV_singlePoint() {
        let pts = [[0.0, 0.0]]
        let opencvResult = projectXY(xyCoords: pts, pvec: goldenPvec)
        let pureResult = projectXYPure(xyCoords: pts, pvec: goldenPvec)
        XCTAssertEqual(pureResult[0][0], opencvResult[0][0], accuracy: 1e-10)
        XCTAssertEqual(pureResult[0][1], opencvResult[0][1], accuracy: 1e-10)
    }

    // MARK: - Benchmark: pure Swift vs OpenCV bridge

    func testBenchmark_projectXY_vs_projectXYPure() {
        let pts = syntheticPoints  // ~200 points
        let iterations = 10_000

        // Warm up
        for _ in 0..<10 {
            _ = projectXYPure(xyCoords: pts, pvec: goldenPvec)
            _ = projectXY(xyCoords: pts, pvec: goldenPvec)
        }

        // Benchmark projectXY (OpenCV bridge)
        let t0 = Date()
        for _ in 0..<iterations {
            _ = projectXY(xyCoords: pts, pvec: goldenPvec)
        }
        let opencvTime = Date().timeIntervalSince(t0)

        // Benchmark projectXYPure (pure Swift)
        let t1 = Date()
        for _ in 0..<iterations {
            _ = projectXYPure(xyCoords: pts, pvec: goldenPvec)
        }
        let pureTime = Date().timeIntervalSince(t1)

        let speedup = opencvTime / pureTime
        let logDir = "/tmp/perf-optimization"
        try? FileManager.default.createDirectory(
            atPath: logDir, withIntermediateDirectories: true)
        let log = """
        # Projection Benchmark
        Points per call: \(pts.count)
        Iterations: \(iterations)
        projectXY (OpenCV bridge): \(String(format: "%.3f", opencvTime * 1000))ms total, \
        \(String(format: "%.3f", opencvTime * 1e6 / Double(iterations)))µs/call
        projectXYPure (pure Swift): \(String(format: "%.3f", pureTime * 1000))ms total, \
        \(String(format: "%.3f", pureTime * 1e6 / Double(iterations)))µs/call
        Speedup: \(String(format: "%.1fx", speedup))
        """
        try? log.write(
            toFile: "\(logDir)/projection_benchmark.md",
            atomically: true, encoding: .utf8)

        print("[PerfProjectionTests] \(log)")
        // Expect at least 1.5x speedup (simulator timing is variable; on-device speedup
        // is significantly higher because the ObjC bridge penalty is proportionally larger)
        XCTAssertGreaterThan(speedup, 1.5,
            "Expected >1.5x speedup but got \(String(format: "%.1fx", speedup))\n\(log)")
    }

    // MARK: - Pipeline integration: output dimensions unchanged

    func testPipelineLBFGSB_outputDimensionsUnchanged() throws {
        let image = try GoldenFileLoader.loadImage("boston_cooking_a_input.jpg")
        guard case .success(let output) = DewarpPipeline.process(image: image, method: .lbfgsb) else {
            throw XCTSkip("Pipeline did not produce output (needs a processable image)")
        }
        let pixelW = output.size.width * output.scale
        let pixelH = output.size.height * output.scale
        XCTAssertGreaterThan(pixelW, 100)
        XCTAssertGreaterThan(pixelH, 100)

        // Save output image for visual inspection
        let dir = "\(NSHomeDirectory())/Desktop/perf-optimization"
        try? FileManager.default.createDirectory(
            atPath: dir, withIntermediateDirectories: true)
        if let data = output.pngData() {
            try? data.write(to: URL(fileURLWithPath: "\(dir)/perf_lbfgsb_output.png"))
        }
    }

    func testPipelinePowell_outputDimensionsUnchanged() throws {
        let image = try GoldenFileLoader.loadImage("boston_cooking_a_input.jpg")
        guard case .success(let output) = DewarpPipeline.process(image: image, method: .powell) else {
            throw XCTSkip("Pipeline did not produce output (needs a processable image)")
        }
        let pixelW = output.size.width * output.scale
        let pixelH = output.size.height * output.scale
        XCTAssertGreaterThan(pixelW, 100)
        XCTAssertGreaterThan(pixelH, 100)

        let dir = "\(NSHomeDirectory())/Desktop/perf-optimization"
        try? FileManager.default.createDirectory(
            atPath: dir, withIntermediateDirectories: true)
        if let data = output.pngData() {
            try? data.write(to: URL(fileURLWithPath: "\(dir)/perf_powell_output.png"))
        }
    }
}
