// CameraMatrix.swift
// Ported from src/page_dewarp/options/k_opt.py

/// Returns the default intrinsic camera matrix using DewarpConfig.focalLength.
/// Ported from k_opt.py:15-32
///
/// Note: Python uses float32 for OpenCV compatibility. Swift uses Double
/// internally; convert to Float32 at the OpenCV bridge boundary.
///
/// - Returns: 3x3 intrinsic camera matrix [[f,0,0],[0,f,0],[0,0,1]]
func cameraMatrix() -> [[Double]] {
    let f = DewarpConfig.focalLength  // 1.2
    return [
        [f, 0, 0],
        [0, f, 0],
        [0, 0, 1],
    ]
}
