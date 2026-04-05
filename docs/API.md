# API Reference

## Overview

PageDewarp is an iOS library that converts warped or curved page photos into flat, thresholded output images. It uses a cubic sheet model to detect curved text lines, fit a 3D surface, and reproject the image onto a flat plane.

```swift
import PageDewarp

let result = DewarpPipeline.process(image: myPhoto)
```

## Public API

### `DewarpPipeline`

The main entry point. A stateless class with a single static method.

```swift
public class DewarpPipeline {
    public static func process(image: UIImage) -> Result<UIImage, DewarpError>
}
```

#### `process(image:)`

Runs the full dewarping pipeline on a page photo.

- **Parameter** `image`: A `UIImage` of any orientation and scale. Works best with photos of book pages or documents where curved text lines are visible.
- **Returns**: `Result<UIImage, DewarpError>`
  - `.success(UIImage)` — A flat, binary-thresholded output image (or grayscale if `noBinary` is set).
  - `.failure(DewarpError)` — One of the error cases below.

The method is synchronous and performs all processing on the calling thread. For UI responsiveness, call it from a background queue.

### `DewarpError`

```swift
public enum DewarpError: Error {
    case noContoursDetected
    case insufficientSpans(count: Int)
    case solvePnPFailed
    case invalidPageDimensions
}
```

| Case | Meaning |
|------|---------|
| `noContoursDetected` | No text or line contours were found in the image. The image may be blank, too noisy, or not contain readable text. |
| `insufficientSpans(count:)` | Contours were found but could not be assembled into enough horizontal spans. `count` is the number found. |
| `solvePnPFailed` | The initial 3D pose estimation (`solvePnP`) did not converge. Usually means the detected keypoints are degenerate. |
| `invalidPageDimensions` | The optimized page dimensions were invalid (e.g., negative). |

## Pipeline Behavior

When you call `process(image:)`, the following steps execute in order:

1. **Resize** — The input is downscaled to fit within 1280x700 pixels (working resolution). The original full-resolution image is used for the final remap.
2. **Page mask** — A rectangular region is defined with margins (50px horizontal, 20px vertical) to exclude edges.
3. **Contour detection (text mode)** — Adaptive thresholding + morphological operations detect text-like contours.
4. **Span assembly** — Contours are chained into horizontal spans based on overlap, distance, and angle constraints.
5. **Line-mode fallback** — If fewer than 3 text spans are found, the pipeline retries with line-detection morphology. The mode producing more spans wins.
6. **Keypoint sampling** — Points are sampled along each span at regular intervals.
7. **3D pose estimation** — `solvePnP` computes initial rotation and translation vectors from the keypoints.
8. **L-BFGS-B optimization** — A cubic surface model is optimized to minimize reprojection error, using analytical gradients.
9. **Page dimension optimization** — Powell minimization refines the output page width and height.
10. **Remap** — The full-resolution input image is warped through the optimized surface model and adaptive-thresholded into a clean binary image.

## Configuration Reference

All configuration values are compile-time constants defined in `DewarpConfig`. They are not currently runtime-configurable; to change them, modify `Sources/PageDewarp/DewarpConfig.swift` and rebuild.

### Optimization

| Constant | Type | Default | Description |
|----------|------|---------|-------------|
| `optMaxIter` | `Int` | `600_000` | Maximum iterations for the L-BFGS-B optimizer. |

### Camera

| Constant | Type | Default | Description |
|----------|------|---------|-------------|
| `focalLength` | `Double` | `1.2` | Focal length in normalized units. Used to construct the camera intrinsic matrix. |

### Text / Contour Detection

| Constant | Type | Default | Description |
|----------|------|---------|-------------|
| `textMinWidth` | `Int` | `15` | Minimum bounding-box width (pixels) for a contour to be considered text. |
| `textMinHeight` | `Int` | `2` | Minimum bounding-box height (pixels) for a contour to be considered text. |
| `textMinAspect` | `Double` | `1.5` | Minimum width/height aspect ratio for text contours. Filters out square or tall blobs. |
| `textMaxThickness` | `Int` | `10` | Maximum column thickness (pixels) for a contour. Rejects thick non-text regions. |

### Span Assembly

| Constant | Type | Default | Description |
|----------|------|---------|-------------|
| `edgeMaxOverlap` | `Double` | `1.0` | Maximum horizontal overlap allowed between adjacent contours in a span. |
| `edgeMaxLength` | `Double` | `100.0` | Maximum distance (pixels) between contour endpoints to be chained into a span. |
| `edgeAngleCost` | `Double` | `10.0` | Weighting factor for angular difference when scoring edges between contours. |
| `edgeMaxAngle` | `Double` | `7.5` | Maximum angle difference (degrees) between contours for them to be chained. |
| `spanMinWidth` | `Int` | `30` | Minimum total width (pixels) for a span to be kept. |
| `spanPxPerStep` | `Int` | `20` | Pixel spacing between sampled keypoints along a span. |

### Image Processing

| Constant | Type | Default | Description |
|----------|------|---------|-------------|
| `screenMaxW` | `Int` | `1280` | Maximum working-resolution width. Input images wider than this are downscaled. |
| `screenMaxH` | `Int` | `700` | Maximum working-resolution height. Input images taller than this are downscaled. |
| `pageMarginX` | `Int` | `50` | Horizontal margin (pixels) excluded from contour detection at left and right edges. |
| `pageMarginY` | `Int` | `20` | Vertical margin (pixels) excluded from contour detection at top and bottom edges. |
| `adaptiveWinsz` | `Int` | `55` | Block size for adaptive thresholding. Must be odd. Larger values tolerate more uneven lighting. |

### Output

| Constant | Type | Default | Description |
|----------|------|---------|-------------|
| `outputZoom` | `Double` | `1.0` | Zoom factor for the output image. `1.0` produces output at half the input height. Values > 1.0 produce larger output. |
| `outputDpi` | `Int` | `300` | Target output DPI (informational; used for dimension calculation). |
| `remapDecimate` | `Int` | `16` | Decimation factor for the coordinate remap grid. The remap is computed on a grid 1/16th the output size, then upscaled. Higher values are faster but less precise. |
| `noBinary` | `Bool` | `false` | When `false` (default), the output is adaptive-thresholded to a black-and-white binary image. When `true`, the output is grayscale (no thresholding). |

### Optimization Tuning

| Constant | Type | Default | Description |
|----------|------|---------|-------------|
| `shearCost` | `Double` | `0.0` | Regularization weight penalizing shear in the cubic surface model. `0.0` means no shear penalty. |
| `maxCorr` | `Int` | `100` | Maximum number of L-BFGS-B correction pairs stored. |

### Parameter Vector Indices

These define the layout of the optimizer's parameter vector. They are internal to the algorithm and not typically modified.

| Constant | Type | Default | Description |
|----------|------|---------|-------------|
| `rvecIdx` | `Range<Int>` | `0..<3` | Indices for the rotation vector (Rodrigues) in the parameter vector. |
| `tvecIdx` | `Range<Int>` | `3..<6` | Indices for the translation vector in the parameter vector. |
| `cubicIdx` | `Range<Int>` | `6..<8` | Indices for the two cubic surface coefficients. |

## Usage Modes

### Library via Swift Package Manager

Add the dependency to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/erykpiast/page-dewarp-swift.git", from: "1.0.0")
]
```

Then in your target:

```swift
.target(name: "YourApp", dependencies: ["PageDewarp"])
```

### Detection Modes

The pipeline operates in two detection modes automatically:

- **Text mode** (default, tried first) — Uses morphological dilation (9x1) and erosion (1x3) to detect text-like horizontal strokes.
- **Line mode** (fallback) — Uses erosion (3x1, 3 iterations) and dilation (8x2) to detect horizontal rules and table lines. Activated automatically when text mode produces fewer than 3 spans.

The mode producing more spans is used. No user intervention is needed.

### Output Format

- **Binary (default)** — The output is adaptive-thresholded to produce a clean black-and-white image suitable for OCR or printing. Controlled by `DewarpConfig.noBinary = false`.
- **Grayscale** — Set `DewarpConfig.noBinary = true` to skip thresholding and get a grayscale remapped image.

## Platform Requirements

- iOS 16.0+
- Swift 5.9+
- Dependencies (vendored, no manual setup needed):
  - **OpenCV 4.10.0** — Minimal build, delivered as a binary XCFramework
  - **L-BFGS-B** — C implementation of the L-BFGS-B optimizer, compiled from source
