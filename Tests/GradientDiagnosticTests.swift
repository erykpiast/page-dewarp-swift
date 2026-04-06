// GradientDiagnosticTests.swift
// Diagnostic test: compare analytical gradient vs finite-difference on IMG_1389.jpeg.
// Identifies which parameter groups have gradient errors.

import XCTest
@testable import PageDewarp
#if SWIFT_PACKAGE
import OpenCVBridge
#endif

class GradientDiagnosticTests: XCTestCase {

    // Helper to build pipeline intermediates from /tmp/IMG_1389.jpeg
    private func buildPipelineState() -> (
        dstpoints: [[Double]],
        keypointIndex: [[Int]],
        spanCounts: [Int],
        initialParams: [Double],
        roughDims: (Double, Double)
    )? {
        let inputPath = "/tmp/IMG_1389.jpeg"
        guard let image = UIImage(contentsOfFile: inputPath) else { return nil }

        let imgH0 = Int(image.size.height * image.scale)
        let imgW0 = Int(image.size.width * image.scale)
        let sclX = Double(imgW0) / Double(DewarpConfig.screenMaxW)
        let sclY = Double(imgH0) / Double(DewarpConfig.screenMaxH)
        let scl = Int(ceil(max(sclX, sclY)))
        let small: UIImage
        if scl > 1 {
            let newW = Int(Double(imgW0) / Double(scl))
            let newH = Int(Double(imgH0) / Double(scl))
            small = OpenCVWrapper.resize(image, width: newW, height: newH, interpolation: 3) ?? image
        } else {
            small = image
        }

        let imgH = Int(small.size.height * small.scale)
        let imgW = Int(small.size.width * small.scale)
        let shape = (height: imgH, width: imgW)

        let xmin = DewarpConfig.pageMarginX, ymin = DewarpConfig.pageMarginY
        let xmax = imgW - xmin, ymax = imgH - ymin
        let pagemask = OpenCVWrapper.createPageMask(
            withWidth: imgW, height: imgH, marginX: xmin, marginY: ymin)
        let pageOutline: [NSValue] = [
            NSValue(cgPoint: CGPoint(x: xmin, y: ymin)),
            NSValue(cgPoint: CGPoint(x: xmin, y: ymax)),
            NSValue(cgPoint: CGPoint(x: xmax, y: ymax)),
            NSValue(cgPoint: CGPoint(x: xmax, y: ymin)),
        ]

        guard let maskImage = OpenCVWrapper.computeDetectionMask(
            small, pagemask: pagemask, isText: true,
            adaptiveWinsz: DewarpConfig.adaptiveWinsz) else { return nil }

        var contours = getContours(maskImage: maskImage)
        var spans = assembleSpans(contours: contours)
        if spans.count < 3 {
            if let lineMask = OpenCVWrapper.computeDetectionMask(
                small, pagemask: pagemask, isText: false,
                adaptiveWinsz: DewarpConfig.adaptiveWinsz) {
                let lc = getContours(maskImage: lineMask)
                let ls = assembleSpans(contours: lc)
                if ls.count > spans.count { contours = lc; spans = ls }
            }
        }
        guard !spans.isEmpty else { return nil }

        let spanPoints = sampleSpans(shape: shape, spans: spans)
        let (corners, ycoords, xcoords) = keypointsFromSamples(
            pagemask: shape, pageOutline: pageOutline, spanPoints: spanPoints)

        guard case .success(let (roughDims, spanCounts, initialParams)) =
            getDefaultParams(corners: corners, ycoords: ycoords, xcoords: xcoords) else { return nil }

        var dstpoints: [[Double]] = [corners[0]]
        for pts in spanPoints { dstpoints.append(contentsOf: pts) }
        let keypointIndex = makeKeypointIndex(spanCounts: spanCounts)

        return (dstpoints, keypointIndex, spanCounts, initialParams, roughDims)
    }

    /// Verify gradient is correct (matches finite differences to 1e-3 relative).
    func testGradientAccuracyOnIMG1389() throws {
        guard let state = buildPipelineState() else {
            print("SKIP: IMG_1389.jpeg not found at /tmp/IMG_1389.jpeg"); return
        }
        let (dstpoints, keypointIndex, spanCounts, initialParams, _) = state

        print("--- Gradient Diagnostic on IMG_1389.jpeg ---")
        print("pvec length: \(initialParams.count), nspans: \(spanCounts.count)")
        print("Initial rvec: \(initialParams[0..<3].map { String(format: "%.6f", $0) })")

        let (f_an, grad_an) = objectiveAndGradient(
            pvec: initialParams, dstpoints: dstpoints, keypointIndex: keypointIndex,
            shearCost: DewarpConfig.shearCost, focalLength: DewarpConfig.focalLength)

        let pureObjective: ([Double]) -> Double = { pvec in
            objectiveAndGradient(pvec: pvec, dstpoints: dstpoints, keypointIndex: keypointIndex,
                shearCost: DewarpConfig.shearCost, focalLength: DewarpConfig.focalLength).f
        }
        let grad_fd_pure = finiteDifferenceGradient(
            objective: pureObjective, x: initialParams, f0: f_an)

        print("Objective: analytical=\(f_an)")

        let nspans = spanCounts.count
        let groups: [(name: String, range: Range<Int>)] = [
            ("rvec",    0..<3),
            ("tvec",    3..<6),
            ("cubic",   6..<8),
            ("ycoords", 8..<(8+nspans)),
            ("xcoords", (8+nspans)..<initialParams.count),
        ]

        var allPass = true
        for grp in groups {
            var maxAbsPure = 0.0, maxRelPure = 0.0
            var worstIdx = grp.range.lowerBound
            for i in grp.range {
                let abs_ = abs(grad_an[i] - grad_fd_pure[i])
                let rel_ = abs_ / max(abs(grad_fd_pure[i]), 1e-8)
                if abs_ > maxAbsPure { maxAbsPure = abs_; worstIdx = i }
                maxRelPure = max(maxRelPure, rel_)
            }
            let status = maxRelPure < 1e-3 ? "OK" : "FAIL"
            print("\(grp.name): maxAbsErr=\(String(format: "%.3e", maxAbsPure)) maxRelErr=\(String(format: "%.3e", maxRelPure)) worstIdx=\(worstIdx) [\(status)]")
            if maxRelPure > 1e-3 { allPass = false }
        }
        print("--- End Gradient Diagnostic ---")
        XCTAssertTrue(allPass, "Gradient has relative errors > 1e-3")
    }

    /// Verify pipeline rvec[0] stays within 0.1 of Python reference (0.015) on IMG_1389.jpeg.
    /// Python uses Powell optimizer; this confirms we're in the same basin.
    func testPipelineRvecCloseToPhythonOnIMG1389() throws {
        let inputPath = "/tmp/IMG_1389.jpeg"
        guard let image = UIImage(contentsOfFile: inputPath) else {
            print("SKIP: IMG_1389.jpeg not found at \(inputPath)"); return
        }

        guard let state = buildPipelineState() else { XCTFail("Pipeline state failed"); return }
        let (dstpoints, keypointIndex, _, initialParams, _) = state

        let objective = makeObjective(
            dstpoints: dstpoints, keypointIndex: keypointIndex,
            shearCost: DewarpConfig.shearCost, rvecRange: DewarpConfig.rvecIdx)

        let optResult = powellMinimize(objective: objective, x0: initialParams)
        let rvec0 = optResult.x[0]
        let pythonRef = 0.015
        print("Powell rvec[0]=\(String(format: "%.6f", rvec0)), Python ref=\(pythonRef), diff=\(String(format: "%.6f", abs(rvec0-pythonRef)))")

        XCTAssertLessThan(abs(rvec0 - pythonRef), 0.1,
            "rvec[0]=\(rvec0) is not within 0.1 of Python reference \(pythonRef)")
    }
}
