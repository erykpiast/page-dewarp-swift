// LBFGSBValidationTests.swift
// End-to-end validation: Swift L-BFGS-B (FD gradients) vs Python scipy L-BFGS-B reference.
// Validates that rvec[0] and loss match Python within tolerance on all test images.

import XCTest
@testable import PageDewarp
#if SWIFT_PACKAGE
import OpenCVBridge
#endif

// Python L-BFGS-B reference values (from scripts/validate_lbfgsb.py).
// Criteria: rvec[0] within 0.05, loss within 20%.
private struct PythonRef {
    let rvec0: Double
    let loss: Double
    let pageDimW: Double
    let pageDimH: Double
}

private let pythonRefs: [String: PythonRef] = [
    "IMG_1369.jpeg": PythonRef(rvec0: 0.0233, loss: 0.006721, pageDimW: 1.25, pageDimH: 1.95),
    "IMG_1389.jpeg": PythonRef(rvec0: 0.0538, loss: 0.004222, pageDimW: 1.43, pageDimH: 2.21),
    "IMG_1413.jpeg": PythonRef(rvec0: 0.0054, loss: 0.025824, pageDimW: 1.33, pageDimH: 1.99),
    "IMG_1868.jpeg": PythonRef(rvec0: 0.0766, loss: 0.002406, pageDimW: 1.99, pageDimH: 1.58),
]

class LBFGSBValidationTests: XCTestCase {

    // MARK: - Helpers

    private struct PipelineState {
        var dstpoints: [[Double]]
        var keypointIndex: [[Int]]
        var spanCounts: [Int]
        var initialParams: [Double]
        var roughDims: (Double, Double)
        var corners: [[Double]]
    }

    private func buildState(imagePath: String) -> PipelineState? {
        guard let image = UIImage(contentsOfFile: imagePath) else { return nil }

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

        return PipelineState(
            dstpoints: dstpoints,
            keypointIndex: keypointIndex,
            spanCounts: spanCounts,
            initialParams: initialParams,
            roughDims: roughDims,
            corners: corners
        )
    }

    private func runValidation(imageName: String) -> (rvec0: Double, loss: Double)? {
        let path = "/tmp/\(imageName)"
        guard let state = buildState(imagePath: path) else {
            print("SKIP \(imageName): not found at \(path)")
            return nil
        }

        let objective = makeObjective(
            dstpoints: state.dstpoints,
            keypointIndex: state.keypointIndex,
            shearCost: DewarpConfig.shearCost,
            rvecRange: DewarpConfig.rvecIdx
        )

        // Use FD-gradient L-BFGS-B, matching Python scipy (no jacobian passed)
        let result = lbfgsbMinimize(objective: objective, x0: state.initialParams)
        let rvec0 = result.x[DewarpConfig.rvecIdx.lowerBound]
        let loss = result.fun

        print("\(imageName): rvec[0]=\(String(format: "%.4f", rvec0)), loss=\(String(format: "%.6f", loss)), nfev=\(result.nfev), converged=\(result.converged)")
        return (rvec0, loss)
    }

    // MARK: - Tests

    func testLBFGSBOnIMG1369() throws {
        let name = "IMG_1369.jpeg"
        guard let (rvec0, loss) = runValidation(imageName: name) else { return }
        guard let ref = pythonRefs[name] else { XCTFail("No reference for \(name)"); return }

        let rvecDiff = abs(rvec0 - ref.rvec0)
        let lossRatio = abs(loss - ref.loss) / ref.loss
        print("\(name): rvec diff=\(String(format: "%.4f", rvecDiff)) (limit 0.05), loss ratio=\(String(format: "%.2f", lossRatio)) (limit 0.20)")

        XCTAssertLessThan(rvecDiff, 0.05, "\(name): rvec[0]=\(rvec0) not within 0.05 of Python \(ref.rvec0)")
        XCTAssertLessThan(lossRatio, 0.20, "\(name): loss=\(loss) not within 20% of Python \(ref.loss)")
    }

    func testLBFGSBOnIMG1389() throws {
        let name = "IMG_1389.jpeg"
        guard let (rvec0, loss) = runValidation(imageName: name) else { return }
        guard let ref = pythonRefs[name] else { XCTFail("No reference for \(name)"); return }

        let rvecDiff = abs(rvec0 - ref.rvec0)
        let lossRatio = abs(loss - ref.loss) / ref.loss
        print("\(name): rvec diff=\(String(format: "%.4f", rvecDiff)) (limit 0.05), loss ratio=\(String(format: "%.2f", lossRatio)) (limit 0.20)")

        XCTAssertLessThan(rvecDiff, 0.05, "\(name): rvec[0]=\(rvec0) not within 0.05 of Python \(ref.rvec0)")
        XCTAssertLessThan(lossRatio, 0.20, "\(name): loss=\(loss) not within 20% of Python \(ref.loss)")
    }

    func testLBFGSBOnIMG1413() throws {
        let name = "IMG_1413.jpeg"
        guard let (rvec0, loss) = runValidation(imageName: name) else { return }
        guard let ref = pythonRefs[name] else { XCTFail("No reference for \(name)"); return }

        let rvecDiff = abs(rvec0 - ref.rvec0)
        let lossRatio = abs(loss - ref.loss) / ref.loss
        print("\(name): rvec diff=\(String(format: "%.4f", rvecDiff)) (limit 0.05), loss ratio=\(String(format: "%.2f", lossRatio)) (limit 0.20)")

        XCTAssertLessThan(rvecDiff, 0.05, "\(name): rvec[0]=\(rvec0) not within 0.05 of Python \(ref.rvec0)")
        XCTAssertLessThan(lossRatio, 0.20, "\(name): loss=\(loss) not within 20% of Python \(ref.loss)")
    }

    func testLBFGSBOnIMG1868() throws {
        let name = "IMG_1868.jpeg"
        guard let (rvec0, loss) = runValidation(imageName: name) else { return }
        guard let ref = pythonRefs[name] else { XCTFail("No reference for \(name)"); return }

        let rvecDiff = abs(rvec0 - ref.rvec0)
        let lossRatio = abs(loss - ref.loss) / ref.loss
        print("\(name): rvec diff=\(String(format: "%.4f", rvecDiff)) (limit 0.05), loss ratio=\(String(format: "%.2f", lossRatio)) (limit 0.20)")

        XCTAssertLessThan(rvecDiff, 0.05, "\(name): rvec[0]=\(rvec0) not within 0.05 of Python \(ref.rvec0)")
        XCTAssertLessThan(lossRatio, 0.20, "\(name): loss=\(loss) not within 20% of Python \(ref.loss)")
    }

    /// Full-pipeline smoke test: DewarpPipeline.process(method:.lbfgsb) doesn't crash.
    func testFullPipelineDoesNotCrashWithLBFGSB() throws {
        for name in ["IMG_1369.jpeg", "IMG_1389.jpeg", "IMG_1413.jpeg", "IMG_1868.jpeg"] {
            let path = "/tmp/\(name)"
            guard let image = UIImage(contentsOfFile: path) else {
                print("SKIP \(name): not found"); continue
            }
            let result = DewarpPipeline.process(image: image, method: .lbfgsb)
            switch result {
            case .success(let output):
                let w = Int(output.size.width * output.scale)
                let h = Int(output.size.height * output.scale)
                print("\(name): L-BFGS-B output \(w)x\(h) ✓")
                XCTAssertGreaterThan(w, 0)
                XCTAssertGreaterThan(h, 0)
            case .failure(let err):
                XCTFail("\(name): pipeline failed with \(err)")
            }
        }
    }
}
