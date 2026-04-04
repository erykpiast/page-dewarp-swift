// ContourDetector.swift
// Ported from src/page_dewarp/contours.py

import Foundation
import UIKit
#if SWIFT_PACKAGE
import OpenCVBridge
#endif

// MARK: - ContourInfo

/// Holds geometric and orientation data about a single text contour.
/// Ported from contours.py:108-168 (ContourInfo class)
final class ContourInfo {
    /// Raw contour points as CGPoint-wrapped NSValues (from OpenCV bridge).
    let contour: [NSValue]
    /// Bounding rectangle (x, y, width, height) in pixel space.
    let rect: CGRect
    /// Centroid [x, y] in pixel space.
    let center: [Double]
    /// Principal axis direction (unit vector from SVD) [x, y].
    let tangent: [Double]
    /// Orientation angle in radians: atan2(tangent.y, tangent.x).
    let angle: Double
    /// Projected extent (min, max) of contour points along the tangent axis.
    let localXRange: (Double, Double)
    /// Point on tangent line at localMin: center + tangent * localMin.
    let point0: [Double]
    /// Point on tangent line at localMax: center + tangent * localMax.
    let point1: [Double]
    /// Previous contour in span chain (set during span assembly).
    var pred: ContourInfo?
    /// Next contour in span chain (set during span assembly).
    var succ: ContourInfo?

    /// Ported from contours.py:111-138 (ContourInfo.__init__)
    init(contour: [NSValue], rect: CGRect, center: [Double], tangent: [Double]) {
        self.contour = contour
        self.rect = rect
        self.center = center
        self.tangent = tangent
        // Ported from contours.py:131
        self.angle = atan2(tangent[1], tangent[0])

        // Project each contour point onto the tangent axis.
        // Ported from contours.py:133-138
        let clx: [Double] = contour.map { val in
            let p = val.cgPointValue
            let dx = Double(p.x) - center[0]
            let dy = Double(p.y) - center[1]
            return tangent[0] * dx + tangent[1] * dy
        }
        let lxmin = clx.min() ?? 0.0
        let lxmax = clx.max() ?? 0.0
        self.localXRange = (lxmin, lxmax)
        // Ported from contours.py:137-138
        self.point0 = [center[0] + tangent[0] * lxmin, center[1] + tangent[1] * lxmin]
        self.point1 = [center[0] + tangent[0] * lxmax, center[1] + tangent[1] * lxmax]
    }

    /// Scalar projection of a 2D point onto this contour's tangent axis.
    /// Ported from contours.py:149-154 (ContourInfo.proj_x)
    func projX(point: [Double]) -> Double {
        return tangent[0] * (point[0] - center[0]) + tangent[1] * (point[1] - center[1])
    }

    /// Overlap of this contour's local axis range with another contour's projected extent.
    /// Ported from contours.py:156-168 (ContourInfo.local_overlap)
    func localOverlap(other: ContourInfo) -> Double {
        let xmin = projX(point: other.point0)
        let xmax = projX(point: other.point1)
        return intervalMeasureOverlap(intA: localXRange, intB: (xmin, xmax))
    }
}

// MARK: - Free functions

/// Return the overlap length of two 1D intervals (may be negative if no overlap).
/// Ported from contours.py:88-105 (interval_measure_overlap)
func intervalMeasureOverlap(intA: (Double, Double), intB: (Double, Double)) -> Double {
    return min(intA.1, intB.1) - max(intA.0, intB.0)
}

/// Compute centroid and principal orientation of a contour via SVD on moment covariance.
/// Returns (center, tangent) or nil if the contour has zero area.
/// Ported from contours.py:46-85 (blob_mean_and_tangent)
func blobMeanAndTangent(contour: [NSValue]) -> ([Double], [Double])? {
    let moments = OpenCVWrapper.moments(ofContour: contour)
    let area = moments["m00"]!.doubleValue
    guard area != 0.0 else { return nil }

    // Ported from contours.py:69-70
    let meanX = moments["m10"]!.doubleValue / area
    let meanY = moments["m01"]!.doubleValue / area

    // Build 2x2 covariance matrix normalised by area: [[mu20, mu11], [mu11, mu02]]
    // Ported from contours.py:71-75
    let mu20 = moments["mu20"]!.doubleValue / area
    let mu11 = moments["mu11"]!.doubleValue / area
    let mu02 = moments["mu02"]!.doubleValue / area
    let covFlat: [NSNumber] = [mu20, mu11, mu11, mu02].map { NSNumber(value: $0) }

    let svd = OpenCVWrapper.svDecomp(ofMatrix: covFlat, rows: 2, cols: 2)
    // U is 2x2 row-major; tangent = first column = [u[0], u[2]]
    // Ported from contours.py:75-77
    let u = svd["u"]!
    let tangent = [u[0].doubleValue, u[2].doubleValue]

    return ([meanX, meanY], tangent)
}

/// Detect and filter contours in a binary grayscale mask image.
/// Returns ContourInfo objects for contours passing size, aspect, and thickness checks.
/// Ported from contours.py:203-239 (get_contours)
///
/// - Parameter maskImage: Grayscale binary mask image (same as Python's `mask` arg).
/// - Returns: Filtered list of ContourInfo objects.
func getContours(maskImage: UIImage) -> [ContourInfo] {
    // Ported from contours.py:219
    let rawContours = OpenCVWrapper.findContours(inGrayImage: maskImage)
    var result: [ContourInfo] = []

    for contour in rawContours {
        // Ported from contours.py:222-229
        let rect = OpenCVWrapper.boundingRect(ofContour: contour)
        let w = Double(rect.width)
        let h = Double(rect.height)

        if w < Double(DewarpConfig.textMinWidth) { continue }
        if h < Double(DewarpConfig.textMinHeight) { continue }
        if w < DewarpConfig.textMinAspect * h { continue }

        // Thickness check BEFORE moments/SVD (matches Python contours.py:230-231 order).
        // Cheap mask-based filter; skip expensive SVD for thick blobs.
        let maxColThickness = OpenCVWrapper.maxColumnThickness(ofContour: contour, boundingRect: rect)
        if maxColThickness > DewarpConfig.textMaxThickness { continue }

        // Compute centroid and principal orientation via moments+SVD.
        // Ported from contours.py:46-85 (blob_mean_and_tangent)
        guard let (center, tangent) = blobMeanAndTangent(contour: contour) else { continue }

        // Create ContourInfo (computes localXRange in init).
        let info = ContourInfo(contour: contour, rect: rect, center: center, tangent: tangent)
        result.append(info)
    }

    return result
}
