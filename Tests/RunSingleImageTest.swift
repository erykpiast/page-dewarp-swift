import XCTest
@testable import PageDewarp
#if SWIFT_PACKAGE
import OpenCVBridge
#endif

class RunSingleImageTest: XCTestCase {
    func testProcessImage() throws {
        let inputPath = "/tmp/input.jpeg"
        guard let image = UIImage(contentsOfFile: inputPath) else {
            XCTFail("Cannot load \(inputPath) — copy your image there first")
            return
        }

        // Read options from /tmp/dewarp_options.txt (one key=value per line)
        var method: DewarpPipeline.OptimizationMethod = .powell
        var output: DewarpPipeline.OutputMode = .color
        if let opts = try? String(contentsOfFile: "/tmp/dewarp_options.txt", encoding: .utf8) {
            for line in opts.split(separator: "\n") {
                let parts = line.split(separator: "=", maxSplits: 1)
                guard parts.count == 2 else { continue }
                switch String(parts[0]) {
                case "method":    if parts[1] == "lbfgsb" { method = .lbfgsb }
                case "binary": if parts[1] == "1" { output = .binary }
                default: break
                }
            }
        }

        let methodName = method == .lbfgsb ? "L-BFGS-B" : "Powell"
        let outputName = output == .binary ? "binary" : "color"

        let t0 = CFAbsoluteTimeGetCurrent()
        let result = DewarpPipeline.process(image: image, method: method, output: output)
        let elapsed = CFAbsoluteTimeGetCurrent() - t0

        switch result {
        case .success(let img):
            let w = Int(img.size.width * img.scale)
            let h = Int(img.size.height * img.scale)
            print("\(methodName) (\(outputName)): \(w)x\(h) in \(String(format: "%.2f", elapsed))s")
            if let data = img.pngData() {
                try data.write(to: URL(fileURLWithPath: "/tmp/output.png"))
            }
        case .failure(let error):
            print("\(methodName) failed: \(error)")
        }
    }
}
