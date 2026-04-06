// DewarpPipeline.swift
// Ported from src/page_dewarp/image.py (WarpedImage class) and mask.py

import Foundation
import UIKit
#if SWIFT_PACKAGE
import OpenCVBridge
#endif

// MARK: - Pipeline

/// Main pipeline orchestrator for page dewarping.
///
/// Wires together contour detection, span assembly, keypoint sampling,
/// parameter optimization, page-dim optimization, and image remapping.
///
/// Ported from image.py:69-259 (WarpedImage)
public class DewarpPipeline {

    // MARK: - Error types

    public enum DewarpError: Error {
        case noContoursDetected
        case insufficientSpans(count: Int)
        case solvePnPFailed
        case invalidPageDimensions
    }

    /// Optimization method for parameter fitting.
    /// Ported from Python's `OPT_METHOD` config / `-m` CLI flag.
    public enum OptimizationMethod {
        /// Powell's conjugate direction method (derivative-free).
        /// Default — matches Python's default and finds the correct basin
        /// on the non-convex page dewarping objective.
        case powell
        /// L-BFGS-B with finite-difference gradients, matching Python's scipy.minimize(method='L-BFGS-B')
        /// which does not pass a jacobian. Uses the same basin as Python for correct convergence.
        case lbfgsb
    }

    // MARK: - Timing breakdown

    /// Timing breakdown for each phase of the pipeline.
    struct TimingBreakdown {
        var preOptimization: TimeInterval   // image load → initial params
        var optimizer: TimeInterval         // lbfgsbMinimize / powellMinimize only
        var postOptimization: TimeInterval  // page dims + remap
        var total: TimeInterval { preOptimization + optimizer + postOptimization }
    }

    // MARK: - Public API

    /// Run the full dewarp pipeline on a UIImage.
    ///
    /// Ported from image.py:79-137 (WarpedImage.__init__)
    ///
    /// - Parameters:
    ///   - image: Input page photo (any orientation, any scale).
    ///   - method: Optimization method to use. Default is `.powell`.
    /// - Returns: `.success(UIImage)` with the dewarped output, or `.failure(DewarpError)`.
    public static func process(
        image: UIImage,
        method: OptimizationMethod = .powell
    ) -> Result<UIImage, DewarpError> {
        // Step 1: resize to screen size if needed.
        // Ported from image.py:92-96 via resize_to_screen
        let small = resizeToScreen(image: image)
        let imgH = Int(small.size.height * small.scale)
        let imgW = Int(small.size.width * small.scale)
        let shape = (height: imgH, width: imgW)

        // Step 2: create page mask with margins.
        // Ported from image.py:98 → calculate_page_extents
        let (pagemask, pageOutline) = makePageExtents(shape: shape)

        // Step 3: detect contours (text mode first).
        // Ported from image.py:99 → contour_info(text=True)
        var contours = detectContours(small: small, pagemask: pagemask, isText: true)

        // Step 4: assemble spans.
        // Ported from image.py:100 → iteratively_assemble_spans
        var spans = assembleSpans(contours: contours)

        // Step 5: fallback to line detection if fewer than 3 spans.
        // Ported from image.py:157-170 (iteratively_assemble_spans)
        if spans.count < 3 {
            let lineContours = detectContours(small: small, pagemask: pagemask, isText: false)
            let lineSpans = assembleSpans(contours: lineContours)
            if lineSpans.count > spans.count {
                contours = lineContours
                spans = lineSpans
            }
        }

        // Step 6: fail if no usable spans.
        // Ported from image.py:102-104
        if spans.isEmpty {
            return .failure(.noContoursDetected)
        }

        // Step 7: sample span keypoints.
        // Ported from image.py:106-108
        let spanPoints = sampleSpans(shape: shape, spans: spans)

        // Step 8: compute corners and per-span coordinates from PCA.
        // Ported from image.py:110-116
        let (corners, ycoords, xcoords) = keypointsFromSamples(
            pagemask: shape,
            pageOutline: pageOutline,
            spanPoints: spanPoints
        )

        // Step 9: get initial params via solvePnP.
        // Ported from image.py:117-121
        let solverResult = getDefaultParams(corners: corners, ycoords: ycoords, xcoords: xcoords)
        guard case .success(let (roughDims, spanCounts, initialParams)) = solverResult else {
            return .failure(.solvePnPFailed)
        }

        // Step 10: build dstpoints = [corners[0]] + flatten(spanPoints).
        // Ported from image.py:122 — np.vstack((corners[0].reshape((1,1,2)),) + tuple(span_points))
        var dstpoints: [[Double]] = [corners[0]]
        for pts in spanPoints {
            dstpoints.append(contentsOf: pts)
        }

        // Step 11: build keypoint index and objective function.
        // Ported from image.py:123-130 → optimise_params
        let keypointIndex = makeKeypointIndex(spanCounts: spanCounts)
        let objective = makeObjective(
            dstpoints: dstpoints,
            keypointIndex: keypointIndex,
            shearCost: DewarpConfig.shearCost,
            rvecRange: DewarpConfig.rvecIdx
        )

        // Step 12: run optimizer.
        // Ported from image.py:123-130 → optimise_params, dispatched by method.
        let optResult: OptimizeResult
        switch method {
        case .powell:
            optResult = powellMinimize(objective: objective, x0: initialParams)
        case .lbfgsb:
            // Use FD gradients to match Python scipy.minimize(method='L-BFGS-B'),
            // which does not pass a jacobian — confirmed root cause of basin divergence.
            optResult = lbfgsbMinimize(objective: objective, x0: initialParams)
        }
        let params = optResult.x

        // Step 13: optimize page dimensions.
        // Ported from image.py:131 → get_page_dims
        var pageDims = getPageDims(corners: corners, roughDims: roughDims, params: params)

        // Step 14: fallback to rough dims if any dimension is negative.
        // Ported from image.py:132-135
        if pageDims[0] < 0 || pageDims[1] < 0 {
            pageDims = [roughDims.0, roughDims.1]
        }

        // Step 15: remap image.
        // Ported from image.py:136 → threshold
        let remap = RemappedImage(img: image, pageDims: pageDims, pvec: params)
        return .success(remap.outputImage)
    }

    /// Run the full dewarp pipeline and return timing breakdown for each phase.
    ///
    /// Identical to `process(image:method:)` but also records wall-clock time for:
    /// - pre-optimization (steps 1–11)
    /// - optimizer-only (step 12)
    /// - post-optimization (steps 13–15)
    ///
    /// Used by performance tests to isolate optimizer overhead from pipeline overhead.
    static func processWithTimingBreakdown(
        image: UIImage,
        method: OptimizationMethod = .powell
    ) -> (result: Result<UIImage, DewarpError>, timing: TimingBreakdown) {
        let t0 = CFAbsoluteTimeGetCurrent()

        let small = resizeToScreen(image: image)
        let imgH = Int(small.size.height * small.scale)
        let imgW = Int(small.size.width * small.scale)
        let shape = (height: imgH, width: imgW)

        let (pagemask, pageOutline) = makePageExtents(shape: shape)

        var contours = detectContours(small: small, pagemask: pagemask, isText: true)
        var spans = assembleSpans(contours: contours)
        if spans.count < 3 {
            let lineContours = detectContours(small: small, pagemask: pagemask, isText: false)
            let lineSpans = assembleSpans(contours: lineContours)
            if lineSpans.count > spans.count {
                contours = lineContours
                spans = lineSpans
            }
        }
        if spans.isEmpty {
            return (.failure(.noContoursDetected),
                    TimingBreakdown(preOptimization: 0, optimizer: 0, postOptimization: 0))
        }

        let spanPoints = sampleSpans(shape: shape, spans: spans)
        let (corners, ycoords, xcoords) = keypointsFromSamples(
            pagemask: shape, pageOutline: pageOutline, spanPoints: spanPoints)
        let solverResult = getDefaultParams(corners: corners, ycoords: ycoords, xcoords: xcoords)
        guard case .success(let (roughDims, spanCounts, initialParams)) = solverResult else {
            return (.failure(.solvePnPFailed),
                    TimingBreakdown(preOptimization: 0, optimizer: 0, postOptimization: 0))
        }

        var dstpoints: [[Double]] = [corners[0]]
        for pts in spanPoints { dstpoints.append(contentsOf: pts) }
        let keypointIndex = makeKeypointIndex(spanCounts: spanCounts)
        let objective = makeObjective(
            dstpoints: dstpoints,
            keypointIndex: keypointIndex,
            shearCost: DewarpConfig.shearCost,
            rvecRange: DewarpConfig.rvecIdx
        )

        let t1 = CFAbsoluteTimeGetCurrent()

        let optResult: OptimizeResult
        switch method {
        case .powell:
            optResult = powellMinimize(objective: objective, x0: initialParams)
        case .lbfgsb:
            optResult = lbfgsbMinimize(objective: objective, x0: initialParams)
        }
        let params = optResult.x

        let t2 = CFAbsoluteTimeGetCurrent()

        var pageDims = getPageDims(corners: corners, roughDims: roughDims, params: params)
        if pageDims[0] < 0 || pageDims[1] < 0 { pageDims = [roughDims.0, roughDims.1] }
        let remap = RemappedImage(img: image, pageDims: pageDims, pvec: params)

        let t3 = CFAbsoluteTimeGetCurrent()

        let timing = TimingBreakdown(
            preOptimization: t1 - t0,
            optimizer: t2 - t1,
            postOptimization: t3 - t2
        )
        return (.success(remap.outputImage), timing)
    }
}

// MARK: - Private helpers

/// Resize an image to fit within SCREEN_MAX_W × SCREEN_MAX_H using INTER_AREA.
/// If the image already fits, return it unchanged.
/// Ported from image.py:200-223 (WarpedImage.resize_to_screen)
private func resizeToScreen(image: UIImage) -> UIImage {
    let imgH = Int(image.size.height * image.scale)
    let imgW = Int(image.size.width * image.scale)
    let sclX = Double(imgW) / Double(DewarpConfig.screenMaxW)
    let sclY = Double(imgH) / Double(DewarpConfig.screenMaxH)
    let scl = Int(ceil(max(sclX, sclY)))
    guard scl > 1 else { return image }
    let invScl = 1.0 / Double(scl)
    let newW = Int(Double(imgW) * invScl)
    let newH = Int(Double(imgH) * invScl)
    // INTER_AREA = 3 in OpenCV
    return OpenCVWrapper.resize(image, width: newW, height: newH, interpolation: 3) ?? image
}

/// Create a page mask (white-filled rectangle, black margins) and the page outline.
/// Returns (pagemaskUIImage, pageOutlineNSValues).
/// Ported from image.py:225-235 (calculate_page_extents)
private func makePageExtents(shape: (height: Int, width: Int)) -> (UIImage, [NSValue]) {
    let width = shape.width
    let height = shape.height
    let xmin = DewarpConfig.pageMarginX
    let ymin = DewarpConfig.pageMarginY
    let xmax = width - xmin
    let ymax = height - ymin

    // Use OpenCV bridge to draw white rectangle on black background.
    // Ported from image.py:231-232 — cv2.rectangle(pagemask, (xmin,ymin), (xmax,ymax), 255, -1)
    let pagemask = OpenCVWrapper.createPageMask(
        withWidth: width, height: height,
        marginX: DewarpConfig.pageMarginX, marginY: DewarpConfig.pageMarginY
    )

    // Page outline corners: TL, BL, BR, TR
    // Ported from image.py:233-236 — page_outline = [[xmin,ymin],[xmin,ymax],[xmax,ymax],[xmax,ymin]]
    let pageOutline: [NSValue] = [
        NSValue(cgPoint: CGPoint(x: xmin, y: ymin)),
        NSValue(cgPoint: CGPoint(x: xmin, y: ymax)),
        NSValue(cgPoint: CGPoint(x: xmax, y: ymax)),
        NSValue(cgPoint: CGPoint(x: xmax, y: ymin)),
    ]

    return (pagemask, pageOutline)
}

/// Detect contours in the image using text or line mode.
/// Ported from image.py:247-259 (contour_info) + mask.py (Mask.calculate)
private func detectContours(small: UIImage, pagemask: UIImage, isText: Bool) -> [ContourInfo] {
    guard let maskImage = OpenCVWrapper.computeDetectionMask(
        small,
        pagemask: pagemask,
        isText: isText,
        adaptiveWinsz: DewarpConfig.adaptiveWinsz
    ) else { return [] }
    return getContours(maskImage: maskImage)
}

/// Optimize final page dimensions using Powell on the bottom-right corner projection error.
/// Ported from image.py:40-66 (get_page_dims)
///
/// - Parameters:
///   - corners: 4 normalized corner points.
///   - roughDims: Initial (width, height) estimate.
///   - params: Optimized parameter vector.
/// - Returns: [width, height] optimized page dimensions.
private func getPageDims(
    corners: [[Double]],
    roughDims: (Double, Double),
    params: [Double]
) -> [Double] {
    // bottom-right corner = corners[2], flattened
    // Ported from image.py:56
    let dstBR = corners[2]

    var dims = [roughDims.0, roughDims.1]

    let dimObjective: ([Double]) -> Double = { dimsLocal in
        // Ported from image.py:59-61
        let projBR = projectXYPure(xyCoords: [dimsLocal], pvec: params)
        let dx = dstBR[0] - projBR[0][0]
        let dy = dstBR[1] - projBR[0][1]
        return dx * dx + dy * dy
    }

    let result = powellMinimize(objective: dimObjective, x0: dims)
    dims = result.x
    return dims
}
