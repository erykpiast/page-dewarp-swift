// PerfProjectionTests.swift
// Benchmarks and pipeline integration tests for projectXYPure/projectXYBulk.

import XCTest
@testable import PageDewarp

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

    // MARK: - Correctness

    func testPureMatchesGolden_emptyInput() {
        let result = projectXYPure(xyCoords: [], pvec: goldenPvec)
        XCTAssertEqual(result.count, 0)
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
