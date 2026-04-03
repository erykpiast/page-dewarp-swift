// AnalyticalGradientTests.swift

import XCTest
@testable import PageDewarp

final class AnalyticalGradientTests: XCTestCase {

    // Test 6: Cubic polynomial derivatives
    func testCubicDerivatives() {
        let x = 0.3, alpha = 0.15, beta = 0.1
        let h = 1e-7
        let a = alpha + beta, b = -2*alpha - beta, c = alpha

        let z0 = ((a*x + b)*x + c)*x

        // dz/dx
        let z1x = ((a*(x+h) + b)*(x+h) + c)*(x+h)
        let dz_dx_fd = (z1x - z0) / h
        let dz_dx_an = 3*(alpha+beta)*x*x - 2*(2*alpha+beta)*x + alpha
        XCTAssertEqual(dz_dx_an, dz_dx_fd, accuracy: 1e-5, "dz/dx")

        // dz/dalpha
        let a1 = (alpha+h) + beta, b1 = -2*(alpha+h) - beta, c1 = (alpha+h)
        let z1a = ((a1*x + b1)*x + c1)*x
        let dz_da_fd = (z1a - z0) / h
        let dz_da_an = x*x*x - 2*x*x + x
        XCTAssertEqual(dz_da_an, dz_da_fd, accuracy: 1e-5, "dz/dalpha")

        // dz/dbeta
        let a2 = alpha + (beta+h), b2 = -2*alpha - (beta+h)
        let z1b = ((a2*x + b2)*x + c)*x
        let dz_db_fd = (z1b - z0) / h
        let dz_db_an = x*x*x - x*x
        XCTAssertEqual(dz_db_an, dz_db_fd, accuracy: 1e-5, "dz/dbeta")
    }

    // Test 7: Clamp boundary gradient is zero
    func testClampBoundary() {
        // Build a minimal parameter vector with alpha at +0.5
        var pvec = [Double](repeating: 0.0, count: 20)
        pvec[0] = -0.05; pvec[1] = 0.07; pvec[2] = 0.01  // rvec
        pvec[3] = -0.6; pvec[4] = -0.96; pvec[5] = 1.22   // tvec
        pvec[6] = 0.5   // alpha CLAMPED
        pvec[7] = 0.1   // beta not clamped
        pvec[8] = 0.1   // y
        pvec[9] = 0.2   // x

        let dstpoints = [[0.0, 0.0], [0.1, 0.1]]
        let keypointIndex = [[0, 0], [9, 8]]

        let (_, grad) = objectiveAndGradient(
            pvec: pvec, dstpoints: dstpoints, keypointIndex: keypointIndex,
            shearCost: 0.0, focalLength: 1.2)
        XCTAssertEqual(grad[6], 0.0, accuracy: 1e-15,
            "Clamped alpha gradient should be zero, got \(grad[6])")
    }

    // Test 8: CRITICAL — Full gradient vs finite differences on golden file data
    func testGradientMatchesFiniteDifferences() throws {
        let image = try GoldenFileLoader.loadImage("boston_cooking_a_input.jpg")

        // Reconstruct pipeline intermediates
        let small = OpenCVWrapper.resize(image,
            width: Int(image.size.width * image.scale) / 5,
            height: Int(image.size.height * image.scale) / 5,
            interpolation: 3) ?? image
        let imgH = Int(small.size.height * small.scale)
        let imgW = Int(small.size.width * small.scale)
        let shape = (height: imgH, width: imgW)

        let pagemask = OpenCVWrapper.createPageMask(
            withWidth: imgW, height: imgH,
            marginX: DewarpConfig.pageMarginX, marginY: DewarpConfig.pageMarginY)
        let xmin = DewarpConfig.pageMarginX, ymin = DewarpConfig.pageMarginY
        let xmax = imgW - xmin, ymax = imgH - ymin
        let pageOutline: [NSValue] = [
            NSValue(cgPoint: CGPoint(x: xmin, y: ymin)),
            NSValue(cgPoint: CGPoint(x: xmin, y: ymax)),
            NSValue(cgPoint: CGPoint(x: xmax, y: ymax)),
            NSValue(cgPoint: CGPoint(x: xmax, y: ymin)),
        ]

        guard let maskImage = OpenCVWrapper.computeDetectionMask(
            small, pagemask: pagemask, isText: true,
            adaptiveWinsz: DewarpConfig.adaptiveWinsz) else {
            XCTFail("Mask computation failed"); return
        }

        let contours = getContours(maskImage: maskImage)
        let spans = assembleSpans(contours: contours)
        guard !spans.isEmpty else { XCTFail("No spans"); return }

        let spanPoints = sampleSpans(shape: shape, spans: spans)
        let (corners, ycoords, xcoords) = keypointsFromSamples(
            pagemask: shape, pageOutline: pageOutline, spanPoints: spanPoints)

        guard case .success(let (_, spanCounts, initialParams)) =
            getDefaultParams(corners: corners, ycoords: ycoords, xcoords: xcoords) else {
            XCTFail("solvePnP failed"); return
        }

        var dstpoints: [[Double]] = [corners[0]]
        for pts in spanPoints { dstpoints.append(contentsOf: pts) }
        let keypointIndex = makeKeypointIndex(spanCounts: spanCounts)

        // Analytical gradient
        let (f_an, grad_an) = objectiveAndGradient(
            pvec: initialParams, dstpoints: dstpoints, keypointIndex: keypointIndex,
            shearCost: DewarpConfig.shearCost, focalLength: DewarpConfig.focalLength)

        // Finite-difference gradient (using OpenCV-based objective for ground truth)
        let objective = makeObjective(
            dstpoints: dstpoints, keypointIndex: keypointIndex,
            shearCost: DewarpConfig.shearCost, rvecRange: DewarpConfig.rvecIdx)
        let f_fd = objective(initialParams)
        let grad_fd = finiteDifferenceGradient(
            objective: objective, x: initialParams, f0: f_fd)

        // Objective values should be very close (pure-Swift vs OpenCV projection)
        let fRelDiff = abs(f_an - f_fd) / max(abs(f_fd), 1e-10)
        print("Objective: analytical=\(f_an) fd=\(f_fd) relDiff=\(fRelDiff)")
        XCTAssertLessThan(fRelDiff, 1e-6,
            "Objective value mismatch: analytical=\(f_an) fd=\(f_fd)")

        // Gradient comparison
        XCTAssertEqual(grad_an.count, grad_fd.count,
            "Gradient length mismatch: \(grad_an.count) vs \(grad_fd.count)")

        var maxAbsErr = 0.0
        var maxRelErr = 0.0
        var worstIdx = 0
        for i in 0..<grad_an.count {
            let absErr = abs(grad_an[i] - grad_fd[i])
            let denom = max(abs(grad_fd[i]), 1e-8)
            let relErr = absErr / denom
            if absErr > maxAbsErr {
                maxAbsErr = absErr
                worstIdx = i
            }
            maxRelErr = max(maxRelErr, relErr)
            XCTAssertEqual(grad_an[i], grad_fd[i],
                accuracy: max(1e-4, abs(grad_fd[i]) * 0.02),
                "grad[\(i)]: analytical=\(grad_an[i]) fd=\(grad_fd[i])")
        }

        print("Max absolute gradient error: \(maxAbsErr) at index \(worstIdx)")
        print("Max relative gradient error: \(maxRelErr)")
        print("Gradient length: \(grad_an.count)")
    }

    // Test 9: Gradient is nonzero at initial params
    func testGradientNonzeroAtInitialParams() {
        // Minimal test — just check the gradient has some magnitude
        var pvec = [Double](repeating: 0.0, count: 20)
        pvec[0] = -0.05; pvec[1] = 0.07; pvec[2] = 0.01
        pvec[3] = -0.6; pvec[4] = -0.96; pvec[5] = 1.22
        pvec[6] = 0.19; pvec[7] = 0.14
        pvec[8] = 0.1; pvec[9] = 0.2; pvec[10] = 0.3
        pvec[11] = 0.4; pvec[12] = 0.5

        let dstpoints = [[0.0, 0.0], [0.1, 0.15], [0.2, 0.1], [0.3, 0.2], [0.15, 0.25]]
        let keypointIndex = [[0, 0], [9, 8], [10, 8], [11, 8], [12, 8]]

        let (_, grad) = objectiveAndGradient(
            pvec: pvec, dstpoints: dstpoints, keypointIndex: keypointIndex,
            shearCost: 0.0, focalLength: 1.2)

        let maxGrad = grad.map { abs($0) }.max() ?? 0
        XCTAssertGreaterThan(maxGrad, 0.001,
            "Gradient should be nonzero, max=\(maxGrad)")
    }
}
