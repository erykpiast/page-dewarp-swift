// DewarpConfig.swift
// Ported from src/page_dewarp/options/core.py (Config struct defaults)

/// Hardcoded configuration defaults matching the Python `Config` msgspec Struct.
/// All values are Swift equivalents of the Python defaults.
enum DewarpConfig {
    // MARK: - Optimization
    // Ported from core.py:68 — OPT_MAX_ITER
    static let optMaxIter = 600_000

    // MARK: - Camera
    // Ported from core.py:150 — FOCAL_LENGTH
    static let focalLength: Double = 1.2

    // MARK: - Text / Contour Detection
    // Ported from core.py:153 — TEXT_MIN_WIDTH
    static let textMinWidth = 15
    // Ported from core.py:164 — TEXT_MIN_HEIGHT
    static let textMinHeight = 2
    // Ported from core.py:172 — TEXT_MIN_ASPECT
    static let textMinAspect: Double = 1.5
    // Ported from core.py:182 — TEXT_MAX_THICKNESS
    static let textMaxThickness = 10

    // MARK: - Span Assembly
    // Ported from core.py:204 — EDGE_MAX_OVERLAP
    static let edgeMaxOverlap: Double = 1.0
    // Ported from core.py:209 — EDGE_MAX_LENGTH
    static let edgeMaxLength: Double = 100.0
    // Ported from core.py:214 — EDGE_ANGLE_COST
    static let edgeAngleCost: Double = 10.0
    // Ported from core.py:216 — EDGE_MAX_ANGLE
    static let edgeMaxAngle: Double = 7.5
    // Ported from core.py:345 — SPAN_MIN_WIDTH
    static let spanMinWidth = 30
    // Ported from core.py:347 — SPAN_PX_PER_STEP
    static let spanPxPerStep = 20

    // MARK: - Image Processing
    // Ported from core.py:221 — SCREEN_MAX_W
    static let screenMaxW = 1280
    // Ported from core.py:223 — SCREEN_MAX_H
    static let screenMaxH = 700
    // Ported from core.py:225 — PAGE_MARGIN_X
    static let pageMarginX = 50
    // Ported from core.py:236 — PAGE_MARGIN_Y
    static let pageMarginY = 20
    // Ported from core.py:248 — ADAPTIVE_WINSZ
    static let adaptiveWinsz = 55

    // MARK: - Output
    // Ported from core.py:260 — OUTPUT_ZOOM
    static let outputZoom: Double = 1.0
    // Ported from core.py:273 — OUTPUT_DPI
    static let outputDpi = 300
    // Ported from core.py:275 — REMAP_DECIMATE
    static let remapDecimate = 16
    // Ported from core.py:277 — NO_BINARY (0 = false)
    static let noBinary = false

    // MARK: - Optimization Tuning
    // Ported from core.py:279 — SHEAR_COST
    static let shearCost: Double = 0.0
    // Ported from core.py:301 — MAX_CORR
    static let maxCorr = 100

    // MARK: - Parameter Vector Indices
    // Ported from core.py:321 — RVEC_IDX = (0, 3)
    static let rvecIdx = 0..<3
    // Ported from core.py:326 — TVEC_IDX = (3, 6)
    static let tvecIdx = 3..<6
    // Ported from core.py:331 — CUBIC_IDX = (6, 8)
    static let cubicIdx = 6..<8
}
