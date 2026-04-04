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

    // MARK: - Public API

    /// Run the full dewarp pipeline on a UIImage.
    ///
    /// Ported from image.py:79-137 (WarpedImage.__init__)
    ///
    /// - Parameter image: Input page photo (any orientation, any scale).
    /// - Returns: `.success(UIImage)` with the dewarped output, or `.failure(DewarpError)`.
    public static func process(image: UIImage) -> Result<UIImage, DewarpError> {
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

        // Step 12: run L-BFGS-B optimizer with analytical gradient.
        // Ported from image.py:123-130 → optimise_params
        let gradObjective: ([Double]) -> (f: Double, grad: [Double]) = { pvec in
            objectiveAndGradient(
                pvec: pvec, dstpoints: dstpoints, keypointIndex: keypointIndex,
                shearCost: DewarpConfig.shearCost, focalLength: DewarpConfig.focalLength
            )
        }
        let optResult = lbfgsbMinimize(objectiveAndGradient: gradObjective, x0: initialParams)
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
        let projBR = projectXY(xyCoords: [dimsLocal], pvec: params)
        let dx = dstBR[0] - projBR[0][0]
        let dy = dstBR[1] - projBR[0][1]
        return dx * dx + dy * dy
    }

    let result = powellMinimize(objective: dimObjective, x0: dims)
    dims = result.x
    return dims
}
