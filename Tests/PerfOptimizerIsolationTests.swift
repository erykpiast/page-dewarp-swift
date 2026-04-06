// PerfOptimizerIsolationTests.swift
// Isolates optimizer-only time from the full pipeline to compare fairly with Python.
//
// Python reference (IMG_1389, L-BFGS-B): optimizer ~0.39s, end-to-end ~0.47s

import XCTest
@testable import PageDewarp
#if SWIFT_PACKAGE
import OpenCVBridge
#endif

final class PerfOptimizerIsolationTests: XCTestCase {

    static let outputDir = "/tmp/perf-optimization"

    override class func setUp() {
        super.setUp()
        try? FileManager.default.createDirectory(
            atPath: outputDir, withIntermediateDirectories: true)
    }

    // MARK: - Optimizer isolation

    func testOptimizerIsolation_IMG1389_LBFGSB() throws {
        let inputPath = "/tmp/IMG_1389.jpeg"
        guard let image = UIImage(contentsOfFile: inputPath) else {
            throw XCTSkip("IMG_1389.jpeg not found at \(inputPath) — copy to /tmp first")
        }

        let (result, timing) = DewarpPipeline.processWithTimingBreakdown(
            image: image, method: .lbfgsb)

        guard case .success(let output) = result else {
            XCTFail("Pipeline failed")
            return
        }

        let w = Int(output.size.width * output.scale)
        let h = Int(output.size.height * output.scale)
        print("[OptimizerIsolation] IMG_1389 (L-BFGS-B)")
        print("  Output: \(w)x\(h)")
        print("  Pre-optimization: \(String(format: "%.3f", timing.preOptimization))s")
        print("  Optimizer only:   \(String(format: "%.3f", timing.optimizer))s  ← compare with Python 0.39s")
        print("  Post-optimization:\(String(format: "%.3f", timing.postOptimization))s")
        print("  Total:            \(String(format: "%.3f", timing.total))s")

        // Save output image
        let desktopDir = "\(NSHomeDirectory())/Desktop/perf-optimization"
        try? FileManager.default.createDirectory(atPath: desktopDir, withIntermediateDirectories: true)
        if let data = output.pngData() {
            try? data.write(to: URL(fileURLWithPath: "\(desktopDir)/optimizer_isolation_IMG1389_lbfgsb.png"))
        }

        // Write markdown report
        try saveReport(image: "IMG_1389", method: "L-BFGS-B",
                       dims: "\(w)x\(h)", timing: timing)

        // Sanity: output must be non-trivial
        XCTAssertGreaterThan(w, 100)
        XCTAssertGreaterThan(h, 100)
    }

    func testOptimizerIsolation_IMG1389_Powell() throws {
        let inputPath = "/tmp/IMG_1389.jpeg"
        guard let image = UIImage(contentsOfFile: inputPath) else {
            throw XCTSkip("IMG_1389.jpeg not found at \(inputPath) — copy to /tmp first")
        }

        let (result, timing) = DewarpPipeline.processWithTimingBreakdown(
            image: image, method: .powell)

        guard case .success(let output) = result else {
            XCTFail("Pipeline failed")
            return
        }

        let w = Int(output.size.width * output.scale)
        let h = Int(output.size.height * output.scale)
        print("[OptimizerIsolation] IMG_1389 (Powell)")
        print("  Output: \(w)x\(h)")
        print("  Pre-optimization: \(String(format: "%.3f", timing.preOptimization))s")
        print("  Optimizer only:   \(String(format: "%.3f", timing.optimizer))s")
        print("  Post-optimization:\(String(format: "%.3f", timing.postOptimization))s")
        print("  Total:            \(String(format: "%.3f", timing.total))s")

        let desktopDir = "\(NSHomeDirectory())/Desktop/perf-optimization"
        try? FileManager.default.createDirectory(atPath: desktopDir, withIntermediateDirectories: true)
        if let data = output.pngData() {
            try? data.write(to: URL(fileURLWithPath: "\(desktopDir)/optimizer_isolation_IMG1389_powell.png"))
        }

        XCTAssertGreaterThan(w, 100)
        XCTAssertGreaterThan(h, 100)
    }

    // MARK: - Multi-image sweep

    func testOptimizerIsolation_AllImages_LBFGSB() throws {
        let images = ["IMG_1369", "IMG_1389", "IMG_1413", "IMG_1868"]
        var rows: [String] = []
        var hasAny = false

        for name in images {
            let path = "/tmp/\(name).jpeg"
            guard let image = UIImage(contentsOfFile: path) else {
                print("[OptimizerIsolation] SKIP \(name): not found at \(path)")
                rows.append("| \(name) | SKIP | - | - | - | - |")
                continue
            }
            hasAny = true

            let (result, timing) = DewarpPipeline.processWithTimingBreakdown(
                image: image, method: .lbfgsb)

            switch result {
            case .success(let output):
                let w = Int(output.size.width * output.scale)
                let h = Int(output.size.height * output.scale)
                rows.append("| \(name) | \(w)x\(h) | " +
                    "\(fmt(timing.preOptimization)) | \(fmt(timing.optimizer)) | " +
                    "\(fmt(timing.postOptimization)) | \(fmt(timing.total)) |")
                print("[OptimizerIsolation] \(name): \(w)x\(h) " +
                    "pre=\(fmt(timing.preOptimization))s opt=\(fmt(timing.optimizer))s " +
                    "post=\(fmt(timing.postOptimization))s total=\(fmt(timing.total))s")
            case .failure(let err):
                rows.append("| \(name) | FAILED (\(err)) | - | - | - | - |")
                print("[OptimizerIsolation] \(name): FAILED - \(err)")
            }
        }

        guard hasAny else {
            throw XCTSkip("No test images found in /tmp — copy IMG_*.jpeg files first")
        }

        let report = buildFullReport(rows: rows)
        let reportPath = "\(Self.outputDir)/optimizer_isolation.md"
        try report.write(toFile: reportPath, atomically: true, encoding: .utf8)
        print("[OptimizerIsolation] Report saved to \(reportPath)")
    }

    // MARK: - Helpers

    private func fmt(_ t: TimeInterval) -> String {
        String(format: "%.3f", t)
    }

    private func saveReport(image: String, method: String, dims: String,
                            timing: DewarpPipeline.TimingBreakdown) throws {
        let report = buildFullReport(rows: [
            "| \(image) | \(dims) | \(fmt(timing.preOptimization)) | " +
            "\(fmt(timing.optimizer)) | \(fmt(timing.postOptimization)) | \(fmt(timing.total)) |"
        ])
        let path = "\(Self.outputDir)/optimizer_isolation.md"
        try report.write(toFile: path, atomically: true, encoding: .utf8)
    }

    private func buildFullReport(rows: [String]) -> String {
        var lines: [String] = [
            "# Optimizer Isolation Benchmark",
            "Generated: \(ISO8601DateFormatter().string(from: Date()))",
            "",
            "Python reference (IMG_1389, L-BFGS-B): optimizer ~0.39s, end-to-end ~0.47s",
            "",
            "| Image | Dims | Pre-opt (s) | Optimizer (s) | Post-opt (s) | Total (s) |",
            "|-------|------|-------------|---------------|--------------|-----------|",
        ]
        lines.append(contentsOf: rows)
        return lines.joined(separator: "\n")
    }
}
