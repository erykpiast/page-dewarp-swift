// LBFGSBComparisonTests.swift
// Runs Swift L-BFGS-B on all test images and saves output to Desktop for comparison with Python.

import XCTest
@testable import PageDewarp
#if SWIFT_PACKAGE
import OpenCVBridge
#endif

class LBFGSBComparisonTests: XCTestCase {

    // /tmp is shared between host and simulator on Apple Silicon
    static let outputDir = "/tmp/lbfgsb-comparison"
    static let testImages = ["IMG_1369", "IMG_1389", "IMG_1413", "IMG_1868"]

    override class func setUp() {
        super.setUp()
        try? FileManager.default.createDirectory(
            atPath: outputDir, withIntermediateDirectories: true)
    }

    func testAllImagesLBFGSB() throws {
        var summaryLines: [String] = [
            "# Swift L-BFGS-B Results",
            "Generated: \(ISO8601DateFormatter().string(from: Date()))",
            "",
            "| Image | Output WxH | Time (s) |",
            "|-------|-----------|----------|",
        ]

        for name in Self.testImages {
            let inputPath = "/tmp/\(name).jpeg"
            guard let inputImage = UIImage(contentsOfFile: inputPath) else {
                print("SKIP: Cannot load \(inputPath)")
                summaryLines.append("| \(name) | SKIP | - |")
                continue
            }

            let startTime = CFAbsoluteTimeGetCurrent()
            let result = DewarpPipeline.process(image: inputImage, method: .lbfgsb)
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime

            switch result {
            case .success(let output):
                let w = Int(output.size.width * output.scale)
                let h = Int(output.size.height * output.scale)
                print("\(name): \(w)x\(h) in \(String(format: "%.2f", elapsed))s")

                // Save output image
                let outPath = Self.outputDir + "/swift_\(name).png"
                if let data = output.pngData() {
                    try data.write(to: URL(fileURLWithPath: outPath))
                }

                summaryLines.append("| \(name) | \(w)x\(h) | \(String(format: "%.2f", elapsed)) |")

            case .failure(let error):
                print("\(name): FAILED - \(error)")
                summaryLines.append("| \(name) | FAILED | - |")
            }
        }

        // Write summary
        let summaryPath = Self.outputDir + "/swift_results.md"
        try summaryLines.joined(separator: "\n").write(
            toFile: summaryPath, atomically: true, encoding: .utf8)
        print("\nSummary written to \(summaryPath)")
    }
}
