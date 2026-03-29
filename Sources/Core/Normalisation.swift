/// Pixel-to-normalized (and vice versa) coordinate transformations.
///
/// Ported from normalisation.py

import Foundation

// MARK: - Normalisation functions

/// Convert image-space pixel coordinates to normalized coordinates.
///
/// Normalized space maps the longer image dimension to [-1, +1].
/// The center of the image maps to (0, 0).
///
/// - Parameters:
///   - shape: Image dimensions as (height, width).
///   - pts: Array of [x, y] pixel coordinate pairs.
/// - Returns: Array of [x, y] normalized coordinate pairs.
///
/// Ported from normalisation.py:15-30
func pix2norm(shape: (height: Int, width: Int), pts: [[Double]]) -> [[Double]] {
    let scl = 2.0 / Double(max(shape.height, shape.width))
    let offsetX = Double(shape.width) * 0.5
    let offsetY = Double(shape.height) * 0.5
    return pts.map { pt in
        [(pt[0] - offsetX) * scl, (pt[1] - offsetY) * scl]
    }
}

/// Convert normalized coordinates back to image-space pixel coordinates.
///
/// - Parameters:
///   - shape: Image dimensions as (height, width).
///   - pts: Array of [x, y] normalized coordinate pairs.
///   - asInteger: If true, round to nearest integer pixel.
/// - Returns: Array of [x, y] pixel coordinate pairs.
///
/// Ported from normalisation.py:33-51
func norm2pix(shape: (height: Int, width: Int), pts: [[Double]], asInteger: Bool) -> [[Double]] {
    let scl = Double(max(shape.height, shape.width)) * 0.5
    let offsetX = 0.5 * Double(shape.width)
    let offsetY = 0.5 * Double(shape.height)
    return pts.map { pt in
        let x = pt[0] * scl + offsetX
        let y = pt[1] * scl + offsetY
        if asInteger {
            return [Double(Int(x + 0.5)), Double(Int(y + 0.5))]
        }
        return [x, y]
    }
}
