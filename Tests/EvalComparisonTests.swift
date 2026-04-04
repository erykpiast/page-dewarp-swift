// EvalComparisonTests.swift
// Runs the full pipeline with L-BFGS-B and compares output to Python reference.

import XCTest
@testable import PageDewarp
#if SWIFT_PACKAGE
import OpenCVBridge
#endif

class EvalComparisonTests: XCTestCase {

    /// Run the pipeline on the comparison input image, save the new Swift output,
    /// and compare pixel-by-pixel with the Python reference.
    func testCompareWithPythonReference() throws {
        // Use the comparison/ input image (same image Python reference was generated from)
        let inputPath = "/Users/eryk.napierala/Projects/page-dewarp-swift/comparison/input.jpg"
        guard let inputImage = UIImage(contentsOfFile: inputPath) else {
            XCTFail("Cannot load input image at \(inputPath)")
            return
        }

        // Run pipeline
        let startTime = CFAbsoluteTimeGetCurrent()
        let result = DewarpPipeline.process(image: inputImage)
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime

        guard case .success(let swiftOutput) = result else {
            XCTFail("Pipeline failed: \(result)")
            return
        }

        let swiftW = Int(swiftOutput.size.width * swiftOutput.scale)
        let swiftH = Int(swiftOutput.size.height * swiftOutput.scale)
        print("--- L-BFGS-B Pipeline Results ---")
        print("Processing time: \(String(format: "%.2f", elapsed))s")
        print("Swift output dimensions: \(swiftW) x \(swiftH)")
        print("Input image dimensions: \(Int(inputImage.size.width * inputImage.scale)) x \(Int(inputImage.size.height * inputImage.scale))")

        // Save Swift output for visual inspection
        let outputPath = "/Users/eryk.napierala/Projects/page-dewarp-swift/comparison/swift_lbfgsb.png"
        if let pngData = swiftOutput.pngData() {
            try pngData.write(to: URL(fileURLWithPath: outputPath))
            print("Saved Swift L-BFGS-B output to: \(outputPath)")
        }

        // Load Python reference
        let pythonPath = "/Users/eryk.napierala/Projects/page-dewarp-swift/comparison/python.png"
        guard let pythonImage = UIImage(contentsOfFile: pythonPath) else {
            print("WARNING: Cannot load Python reference at \(pythonPath), skipping comparison")
            return
        }

        let pythonW = Int(pythonImage.size.width * pythonImage.scale)
        let pythonH = Int(pythonImage.size.height * pythonImage.scale)
        print("Python output dimensions: \(pythonW) x \(pythonH)")

        // Load old Swift/Powell output for comparison
        let powellPath = "/Users/eryk.napierala/Projects/page-dewarp-swift/comparison/swift.png"
        let powellImage = UIImage(contentsOfFile: powellPath)
        if let pi = powellImage {
            let pw = Int(pi.size.width * pi.scale)
            let ph = Int(pi.size.height * pi.scale)
            print("Old Swift (Powell) dimensions: \(pw) x \(ph)")
        }

        // Dimension comparison
        let dimMatchW = Double(min(swiftW, pythonW)) / Double(max(swiftW, pythonW)) * 100
        let dimMatchH = Double(min(swiftH, pythonH)) / Double(max(swiftH, pythonH)) * 100
        print("Dimension match: width=\(String(format: "%.1f", dimMatchW))%, height=\(String(format: "%.1f", dimMatchH))%")

        // Pixel comparison (only if same dimensions)
        if swiftW == pythonW && swiftH == pythonH {
            let matchPct = pixelMatch(swiftOutput, pythonImage)
            print("Pixel match with Python: \(String(format: "%.1f", matchPct))%")
            // Report but don't fail — we're gathering data
        } else {
            // Resize to common dimensions and compare
            let commonW = min(swiftW, pythonW)
            let commonH = min(swiftH, pythonH)
            print("Dimensions differ, comparing at common size \(commonW)x\(commonH) (cropped)")
            let matchPct = pixelMatchCropped(swiftOutput, pythonImage, width: commonW, height: commonH)
            print("Pixel match (cropped region): \(String(format: "%.1f", matchPct))%")
        }

        // Also compare against old Powell output if available
        if let powellImg = powellImage, swiftW == Int(powellImg.size.width * powellImg.scale),
           swiftH == Int(powellImg.size.height * powellImg.scale) {
            let powellMatch = pixelMatch(swiftOutput, powellImg)
            print("Pixel match with old Powell output: \(String(format: "%.1f", powellMatch))%")
        }

        print("--- End Results ---")

        // Attach output image to test results
        let attachment = XCTAttachment(image: swiftOutput)
        attachment.name = "swift_lbfgsb_output"
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    /// Standalone diagnostic: run optimizer on the golden file and print
    /// convergence details and parameter comparison.
    func testOptimizerDiagnostics() throws {
        let image = try GoldenFileLoader.loadImage("boston_cooking_a_input.jpg")

        // We need to access intermediate results. Re-run the pipeline steps manually.
        let small = resizeForDiag(image: image)
        let imgH = Int(small.size.height * small.scale)
        let imgW = Int(small.size.width * small.scale)
        let shape = (height: imgH, width: imgW)
        print("--- Optimizer Diagnostics ---")
        print("Resized image: \(imgW) x \(imgH)")

        let (pagemask, pageOutline) = makePageExtentsForDiag(shape: shape)
        var contours = getContours(maskImage: OpenCVWrapper.computeDetectionMask(
            small, pagemask: pagemask, isText: true, adaptiveWinsz: DewarpConfig.adaptiveWinsz)!)
        var spans = assembleSpans(contours: contours)

        if spans.count < 3 {
            let lineContours = getContours(maskImage: OpenCVWrapper.computeDetectionMask(
                small, pagemask: pagemask, isText: false, adaptiveWinsz: DewarpConfig.adaptiveWinsz)!)
            let lineSpans = assembleSpans(contours: lineContours)
            if lineSpans.count > spans.count {
                contours = lineContours
                spans = lineSpans
            }
        }
        print("Spans: \(spans.count)")

        let spanPoints = sampleSpans(shape: shape, spans: spans)
        let (corners, ycoords, xcoords) = keypointsFromSamples(
            pagemask: shape, pageOutline: pageOutline, spanPoints: spanPoints)

        guard case .success(let (roughDims, spanCounts, initialParams)) =
            getDefaultParams(corners: corners, ycoords: ycoords, xcoords: xcoords) else {
            XCTFail("solvePnP failed")
            return
        }
        print("Initial params count: \(initialParams.count)")
        print("Initial rvec: \(initialParams[0..<3].map { String(format: "%.6f", $0) })")
        print("Initial tvec: \(initialParams[3..<6].map { String(format: "%.6f", $0) })")
        print("Rough dims: \(roughDims)")

        var dstpoints: [[Double]] = [corners[0]]
        for pts in spanPoints { dstpoints.append(contentsOf: pts) }
        let keypointIndex = makeKeypointIndex(spanCounts: spanCounts)
        let objective = makeObjective(
            dstpoints: dstpoints, keypointIndex: keypointIndex,
            shearCost: DewarpConfig.shearCost, rvecRange: DewarpConfig.rvecIdx)

        // Run L-BFGS-B
        let startLBFGSB = CFAbsoluteTimeGetCurrent()
        let lbfgsbResult = lbfgsbMinimize(objective: objective, x0: initialParams)
        let lbfgsbTime = CFAbsoluteTimeGetCurrent() - startLBFGSB

        print("\nL-BFGS-B results:")
        print("  Converged: \(lbfgsbResult.converged)")
        print("  Function evals: \(lbfgsbResult.nfev)")
        print("  Final loss: \(String(format: "%.8f", lbfgsbResult.fun))")
        print("  Time: \(String(format: "%.2f", lbfgsbTime))s")
        print("  rvec: \(lbfgsbResult.x[0..<3].map { String(format: "%.6f", $0) })")
        print("  tvec: \(lbfgsbResult.x[3..<6].map { String(format: "%.6f", $0) })")
        print("  cubic: \(lbfgsbResult.x[6..<8].map { String(format: "%.6f", $0) })")

        // Run Powell for comparison
        let startPowell = CFAbsoluteTimeGetCurrent()
        let powellResult = powellMinimize(objective: objective, x0: initialParams)
        let powellTime = CFAbsoluteTimeGetCurrent() - startPowell

        print("\nPowell results (for comparison):")
        print("  Converged: \(powellResult.converged)")
        print("  Function evals: \(powellResult.nfev)")
        print("  Final loss: \(String(format: "%.8f", powellResult.fun))")
        print("  Time: \(String(format: "%.2f", powellTime))s")
        print("  rvec: \(powellResult.x[0..<3].map { String(format: "%.6f", $0) })")
        print("  tvec: \(powellResult.x[3..<6].map { String(format: "%.6f", $0) })")
        print("  cubic: \(powellResult.x[6..<8].map { String(format: "%.6f", $0) })")

        // Load Python reference
        let golden: GoldenParams = try GoldenFileLoader.loadJSON("optimized_params")
        print("\nPython reference:")
        print("  Final loss: \(golden.final_loss.map { String(format: "%.8f", $0) } ?? "N/A")")
        print("  rvec: \(golden.rvec.map { String(format: "%.6f", $0) })")
        print("  tvec: \(golden.tvec.map { String(format: "%.6f", $0) })")
        print("  cubic: \(golden.cubic.map { String(format: "%.6f", $0) })")
        print("  page dims: \(golden.page_dims.map { String(format: "%.4f", $0) })")

        // Parameter vector distance
        var pythonParams: [Double] = []
        pythonParams.append(contentsOf: golden.rvec)
        pythonParams.append(contentsOf: golden.tvec)
        pythonParams.append(contentsOf: golden.cubic)
        pythonParams.append(contentsOf: golden.ycoords)
        for xc in golden.xcoords { pythonParams.append(contentsOf: xc) }

        if pythonParams.count == lbfgsbResult.x.count {
            var lbfgsbDist = 0.0, powellDist = 0.0
            for i in 0..<pythonParams.count {
                let ld = lbfgsbResult.x[i] - pythonParams[i]
                let pd = powellResult.x[i] - pythonParams[i]
                lbfgsbDist += ld * ld
                powellDist += pd * pd
            }
            print("\nParameter distance from Python (L2):")
            print("  L-BFGS-B: \(String(format: "%.6f", sqrt(lbfgsbDist)))")
            print("  Powell:   \(String(format: "%.6f", sqrt(powellDist)))")
        }

        print("--- End Diagnostics ---")
    }

    // Minimal helpers to replicate pipeline steps without full DewarpPipeline
    private func resizeForDiag(image: UIImage) -> UIImage {
        let imgH = Int(image.size.height * image.scale)
        let imgW = Int(image.size.width * image.scale)
        let sclX = Double(imgW) / Double(DewarpConfig.screenMaxW)
        let sclY = Double(imgH) / Double(DewarpConfig.screenMaxH)
        let scl = Int(ceil(max(sclX, sclY)))
        guard scl > 1 else { return image }
        let invScl = 1.0 / Double(scl)
        let newW = Int(Double(imgW) * invScl)
        let newH = Int(Double(imgH) * invScl)
        return OpenCVWrapper.resize(image, width: newW, height: newH, interpolation: 3) ?? image
    }

    private func makePageExtentsForDiag(shape: (height: Int, width: Int)) -> (UIImage, [NSValue]) {
        let xmin = DewarpConfig.pageMarginX
        let ymin = DewarpConfig.pageMarginY
        let xmax = shape.width - xmin
        let ymax = shape.height - ymin
        let pagemask = OpenCVWrapper.createPageMask(
            withWidth: shape.width, height: shape.height,
            marginX: DewarpConfig.pageMarginX, marginY: DewarpConfig.pageMarginY)
        let pageOutline: [NSValue] = [
            NSValue(cgPoint: CGPoint(x: xmin, y: ymin)),
            NSValue(cgPoint: CGPoint(x: xmin, y: ymax)),
            NSValue(cgPoint: CGPoint(x: xmax, y: ymax)),
            NSValue(cgPoint: CGPoint(x: xmax, y: ymin)),
        ]
        return (pagemask, pageOutline)
    }

    // MARK: - Pixel comparison helpers

    /// Compare two images pixel-by-pixel (must be same dimensions).
    /// Returns percentage of matching pixels (within tolerance of 1/255).
    private func pixelMatch(_ a: UIImage, _ b: UIImage) -> Double {
        guard let aData = a.cgImage?.dataProvider?.data as Data?,
              let bData = b.cgImage?.dataProvider?.data as Data? else {
            return 0
        }

        let count = min(aData.count, bData.count)
        guard count > 0 else { return 0 }

        var matches = 0
        aData.withUnsafeBytes { aPtr in
            bData.withUnsafeBytes { bPtr in
                let aBytes = aPtr.bindMemory(to: UInt8.self)
                let bBytes = bPtr.bindMemory(to: UInt8.self)
                for i in 0..<count {
                    let diff = Int(aBytes[i]) - Int(bBytes[i])
                    if abs(diff) <= 1 {
                        matches += 1
                    }
                }
            }
        }

        return Double(matches) / Double(count) * 100
    }

    /// Compare two images by cropping to a common region.
    private func pixelMatchCropped(_ a: UIImage, _ b: UIImage, width: Int, height: Int) -> Double {
        // Render both images at the common size
        let size = CGSize(width: width, height: height)
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1.0
        let renderer = UIGraphicsImageRenderer(size: size, format: format)

        let aRendered = renderer.image { ctx in
            a.draw(in: CGRect(origin: .zero, size: size))
        }
        let bRendered = renderer.image { ctx in
            b.draw(in: CGRect(origin: .zero, size: size))
        }

        return pixelMatch(aRendered, bRendered)
    }
}
