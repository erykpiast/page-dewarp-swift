// SpanAssembler.swift
// Ported from src/page_dewarp/spans.py

import Foundation
#if SWIFT_PACKAGE
import OpenCVBridge
#endif

// MARK: - Angle distance

/// Signed angular distance between two angles, normalized to [-π, π], then absolute value.
/// Ported from spans.py:33-43
func angleDist(_ angleB: Double, _ angleA: Double) -> Double {
    var diff = angleB - angleA
    while diff > .pi { diff -= 2 * .pi }
    while diff < -.pi { diff += 2 * .pi }
    return abs(diff)
}

// MARK: - Candidate edge generation

/// Compute a candidate edge (score, left, right) between two contours, or nil if rejected.
/// Score = distance + edgeAngleCost * deltaAngle.
/// Ported from spans.py:46-83
func generateCandidateEdge(
    _ cinfoA: ContourInfo,
    _ cinfoB: ContourInfo
) -> (Double, ContourInfo, ContourInfo)? {
    var a = cinfoA
    var b = cinfoB
    // Ensure a is to the left of b (swap if needed).
    // Ported from spans.py:59-60
    if a.point0[0] > b.point1[0] { swap(&a, &b) }

    let xOverlapA = a.localOverlap(other: b)
    let xOverlapB = b.localOverlap(other: a)

    // Overall direction from a center to b center.
    let overallAngle = atan2(b.center[1] - a.center[1], b.center[0] - a.center[0])

    // Ported from spans.py:64-72
    let deltaAngle = max(
        angleDist(a.angle, overallAngle),
        angleDist(b.angle, overallAngle)
    ) * 180.0 / .pi

    let xOverlap = max(xOverlapA, xOverlapB)
    let dx = b.point0[0] - a.point1[0]
    let dy = b.point0[1] - a.point1[1]
    let dist = sqrt(dx * dx + dy * dy)

    // Ported from spans.py:76-82
    if dist > DewarpConfig.edgeMaxLength
        || xOverlap > DewarpConfig.edgeMaxOverlap
        || deltaAngle > DewarpConfig.edgeMaxAngle {
        return nil
    }
    let score = dist + deltaAngle * DewarpConfig.edgeAngleCost
    return (score, a, b)
}

// MARK: - Span assembly

/// Assemble a flat list of ContourInfo objects into left-to-right chains (spans).
/// Ported from spans.py:86-161
func assembleSpans(contours: [ContourInfo]) -> [[ContourInfo]] {
    // Sort by top y-coordinate (rect[1] in Python).
    // Ported from spans.py:110
    let cinfos = contours.sorted { $0.rect.minY < $1.rect.minY }

    // Generate all O(n^2) candidate edges.
    // Ported from spans.py:111-117
    var candidateEdges: [(Double, ContourInfo, ContourInfo)] = []
    for i in 0..<cinfos.count {
        for j in 0..<i {
            if let edge = generateCandidateEdge(cinfos[i], cinfos[j]) {
                candidateEdges.append(edge)
            }
        }
    }

    // Sort by score ascending (lower = better).
    // Ported from spans.py:120
    candidateEdges.sort { $0.0 < $1.0 }

    // Greedily link: each contour gets at most one successor and one predecessor.
    // Ported from spans.py:126-130
    for (_, cinfoA, cinfoB) in candidateEdges {
        if cinfoA.succ == nil && cinfoB.pred == nil {
            cinfoA.succ = cinfoB
            cinfoB.pred = cinfoA
        }
    }

    // Build chains by walking from each head (no predecessor) to its end.
    // Ported from spans.py:133-156
    var spans: [[ContourInfo]] = []
    var remaining: [ContourInfo] = cinfos

    while !remaining.isEmpty {
        // Walk back from first remaining to the head of its chain.
        var head = remaining[0]
        while let p = head.pred { head = p }

        // Walk forward collecting the chain, removing each element from remaining.
        var curSpan: [ContourInfo] = []
        var width = 0.0
        var cur: ContourInfo? = head
        while let c = cur {
            remaining.removeAll { $0 === c }
            curSpan.append(c)
            width += c.localXRange.1 - c.localXRange.0
            cur = c.succ
        }

        if width > Double(DewarpConfig.spanMinWidth) {
            spans.append(curSpan)
        }
    }

    return spans
}

// MARK: - Keypoint sampling

/// Sample keypoints along text spans at regular horizontal pixel intervals.
/// Returns a list of normalized-coordinate point arrays, one per span.
/// Ported from spans.py:164-198
func sampleSpans(
    shape: (height: Int, width: Int),
    spans: [[ContourInfo]]
) -> [[[Double]]] {
    var spanPoints: [[[Double]]] = []

    for span in spans {
        var contourPoints: [[Double]] = []

        for cinfo in span {
            let xmin = Double(cinfo.rect.origin.x)
            let ymin = Double(cinfo.rect.origin.y)

            // Get vertical mean y per column of the tight contour mask.
            // Ported from spans.py:186-188
            let means = OpenCVWrapper.columnMeans(ofContour: cinfo.contour, boundingRect: cinfo.rect)
            let w = means.count
            guard w > 0 else { continue }

            // Centered start offset.
            // Ported from spans.py:191
            let step = DewarpConfig.spanPxPerStep
            let start = ((w - 1) % step) / 2

            // Ported from spans.py:192-194
            var x = start
            while x < w {
                let meanY = means[x].doubleValue
                contourPoints.append([Double(x) + xmin, meanY + ymin])
                x += step
            }
        }

        guard !contourPoints.isEmpty else { continue }

        // Convert to normalized coordinates.
        // Ported from spans.py:195-196
        let normalized = pix2norm(shape: shape, pts: contourPoints)
        spanPoints.append(normalized)
    }

    return spanPoints
}

// MARK: - Keypoints from samples

/// Compute page-corner keypoints and per-span y/x coordinates from span sample points.
///
/// Returns (corners, ycoords, xcoords):
///   - corners: 4 normalized [x,y] points defining the page rectangle.
///   - ycoords: per-span mean y-coordinate (page-relative).
///   - xcoords: per-span array of x-coordinates (page-relative).
///
/// Ported from spans.py:201-264
func keypointsFromSamples(
    pagemask: (height: Int, width: Int),
    pageOutline: [NSValue],
    spanPoints: [[[Double]]]
) -> (corners: [[Double]], ycoords: [Double], xcoords: [[Double]]) {
    // Weighted PCA over all spans to find the dominant horizontal direction.
    // Ported from spans.py:228-235
    var allEvecsX = 0.0
    var allEvecsY = 0.0
    var allWeights = 0.0

    for points in spanPoints {
        guard points.count >= 2 else { continue }
        let nsPoints = points.map { pt in NSValue(cgPoint: CGPoint(x: pt[0], y: pt[1])) }
        let pca = OpenCVWrapper.pcaCompute(onPoints: nsPoints)
        let evecs = pca["eigenvectors"]!
        // First eigenvector = dominant axis [evecs[0], evecs[1]].
        let evecX = evecs[0].doubleValue
        let evecY = evecs[1].doubleValue

        // Weight by span length (distance from first to last point).
        // Ported from spans.py:232
        let first = points[0], last = points[points.count - 1]
        let weight = sqrt((last[0] - first[0]) * (last[0] - first[0]) +
                          (last[1] - first[1]) * (last[1] - first[1]))

        allEvecsX += evecX * weight
        allEvecsY += evecY * weight
        allWeights += weight
    }

    // Ported from spans.py:235-239
    var xDirX = allWeights > 0 ? allEvecsX / allWeights : 1.0
    var xDirY = allWeights > 0 ? allEvecsY / allWeights : 0.0
    if xDirX < 0 { xDirX = -xDirX; xDirY = -xDirY }
    let yDirX = -xDirY
    let yDirY = xDirX

    // Build page corners from the convex hull of pageOutline.
    // Ported from spans.py:241-254
    let hullValues = OpenCVWrapper.convexHull(ofPoints: pageOutline)
    let hullPts = hullValues.map { v -> [Double] in
        let p = v.cgPointValue
        return [Double(p.x), Double(p.y)]
    }
    let pageCoords = pix2norm(shape: pagemask, pts: hullPts)

    // Project onto x_dir and y_dir, find bounding extents.
    let pxCoords = pageCoords.map { p in p[0] * xDirX + p[1] * xDirY }
    let pyCoords = pageCoords.map { p in p[0] * yDirX + p[1] * yDirY }
    let px0 = pxCoords.min() ?? 0.0
    let px1 = pxCoords.max() ?? 0.0
    let py0 = pyCoords.min() ?? 0.0
    let py1 = pyCoords.max() ?? 0.0

    // Build four corners: (px0,py0), (px1,py0), (px1,py1), (px0,py1).
    // Ported from spans.py:251-254 — x_dir_coeffs=[px0,px1,px1,px0], y_dir_coeffs=[py0,py0,py1,py1]
    let xCoeffs = [px0, px1, px1, px0]
    let yCoeffs = [py0, py0, py1, py1]
    var corners: [[Double]] = []
    for i in 0..<4 {
        corners.append([
            xCoeffs[i] * xDirX + yCoeffs[i] * yDirX,
            xCoeffs[i] * xDirY + yCoeffs[i] * yDirY,
        ])
    }

    // Compute per-span xcoords and ycoords relative to (px0, py0).
    // Ported from spans.py:256-261
    var xcoords: [[Double]] = []
    var ycoords: [Double] = []
    for points in spanPoints {
        let pxC = points.map { p in p[0] * xDirX + p[1] * xDirY }
        let pyC = points.map { p in p[0] * yDirX + p[1] * yDirY }
        xcoords.append(pxC.map { $0 - px0 })
        let meanPy = pyC.reduce(0.0, +) / Double(pyC.count)
        ycoords.append(meanPy - py0)
    }

    return (corners: corners, ycoords: ycoords, xcoords: xcoords)
}
