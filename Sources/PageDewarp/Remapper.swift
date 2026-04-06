// Remapper.swift
// Ported from src/page_dewarp/dewarp.py

import Foundation
import UIKit
#if SWIFT_PACKAGE
import OpenCVBridge
#endif

// MARK: - Helpers

/// Round integer `i` up to the nearest multiple of `factor`.
/// If already a multiple, return unchanged.
/// Ported from dewarp.py:34-42
func roundNearestMultiple(_ i: Int, _ factor: Int) -> Int {
    let rem = i % factor
    return rem == 0 ? i : i + factor - rem
}

/// Generate `count` evenly-spaced values from `start` to `end` (inclusive).
/// Mirrors np.linspace(start, end, count).
private func linspace(_ start: Double, _ end: Double, _ count: Int) -> [Double] {
    guard count > 1 else { return count == 1 ? [start] : [] }
    let step = (end - start) / Double(count - 1)
    return (0..<count).map { start + Double($0) * step }
}

// MARK: - Remapper

/// Rectify and threshold an image based on a cubic page parameterization.
/// Ported from dewarp.py:45-151
struct RemappedImage {
    /// The final output image (adaptive-thresholded binary, or grayscale if noBinary).
    let outputImage: UIImage

    /// - Parameters:
    ///   - img: Full-resolution input UIImage (RGB/RGBA).
    ///   - pageDims: [width, height] of the page in normalized units.
    ///   - pvec: Optimized parameter vector (rvec, tvec, cubic coefficients).
    /// Ported from dewarp.py:57-151
    init(img: UIImage, pageDims: [Double], pvec: [Double]) {
        // Pixel dimensions (accounting for screen scale)
        let imgH = Int(img.size.height * img.scale)
        let imgW = Int(img.size.width * img.scale)
        let imgShape = (height: imgH, width: imgW)

        // Ported from dewarp.py:79-84 — output dimension calculation
        var outHeight = Int(0.5 * pageDims[1] * DewarpConfig.outputZoom * Double(imgH))
        outHeight = roundNearestMultiple(outHeight, DewarpConfig.remapDecimate)
        var outWidth = Int(Double(outHeight) * pageDims[0] / pageDims[1])
        outWidth = roundNearestMultiple(outWidth, DewarpConfig.remapDecimate)

        // Ported from dewarp.py:86-89 — decimated grid dimensions
        let hSmall = outHeight / DewarpConfig.remapDecimate
        let wSmall = outWidth / DewarpConfig.remapDecimate

        // Ported from dewarp.py:90-92 — linspace ranges
        let pageXRange = linspace(0.0, pageDims[0], wSmall)
        let pageYRange = linspace(0.0, pageDims[1], hSmall)

        // Ported from dewarp.py:92-98 — meshgrid → flatten
        // np.meshgrid(x_range, y_range) produces x varying along columns, y along rows.
        // Flattened row-major: outer loop = rows (y), inner loop = cols (x).
        var pageXYCoords: [[Double]] = []
        pageXYCoords.reserveCapacity(hSmall * wSmall)
        for iy in 0..<hSmall {
            for ix in 0..<wSmall {
                pageXYCoords.append([pageXRange[ix], pageYRange[iy]])
            }
        }

        // Ported from dewarp.py:100-103 — project and convert to pixel coords
        // Using projectXYPure (pure Swift, identical math, avoids ObjC bridge overhead)
        let imagePoints = projectXYPure(xyCoords: pageXYCoords, pvec: pvec)
        let imagePixelPoints = norm2pix(shape: imgShape, pts: imagePoints, asInteger: false)

        // Build flat float32 maps without NSNumber boxing
        let totalSmall = hSmall * wSmall
        var mapXFloats = [Float](repeating: 0, count: totalSmall)
        var mapYFloats = [Float](repeating: 0, count: totalSmall)
        for (i, pt) in imagePixelPoints.enumerated() {
            mapXFloats[i] = Float(pt[0])
            mapYFloats[i] = Float(pt[1])
        }
        let mapXSmallData = Data(bytes: mapXFloats, count: totalSmall * MemoryLayout<Float>.size)
        let mapYSmallData = Data(bytes: mapYFloats, count: totalSmall * MemoryLayout<Float>.size)

        // Ported from dewarp.py:105-114 — resize coordinate maps to full output resolution
        let mapXFullData = OpenCVWrapper.resizeFloatMapData(
            mapXSmallData, srcWidth: wSmall, srcHeight: hSmall,
            dstWidth: outWidth, dstHeight: outHeight
        )!
        let mapYFullData = OpenCVWrapper.resizeFloatMapData(
            mapYSmallData, srcWidth: wSmall, srcHeight: hSmall,
            dstWidth: outWidth, dstHeight: outHeight
        )!

        // Ported from dewarp.py:116-125 — remap the grayscale image
        let remapped = OpenCVWrapper.remapImageData(
            img, mapXData: mapXFullData, mapYData: mapYFullData,
            width: outWidth, height: outHeight
        )!

        // Ported from dewarp.py:126-139 — adaptive threshold or passthrough
        if DewarpConfig.noBinary {
            self.outputImage = remapped
        } else {
            let threshed = OpenCVWrapper.adaptiveThresholdImage(
                remapped,
                maxValue: 255.0,
                blockSize: DewarpConfig.adaptiveWinsz,
                c: 25.0
            )!
            self.outputImage = threshed
        }
    }
}
