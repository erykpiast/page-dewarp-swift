# Differences from Python

This document details how the Swift implementation differs from the [Python reference](https://github.com/lemonzi/page-dewarp).

## Module Mapping

| Python | Swift | Notes |
|--------|-------|-------|
| `image.py` (WarpedImage) | `DewarpPipeline.swift` | Static methods instead of class instance |
| `contours.py` | `ContourDetector.swift` | Same algorithm, OpenCV bridge for moments/SVD |
| `spans.py` | `SpanAssembler.swift` + `Keypoints.swift` | Split into assembly and sampling |
| `keypoints.py` | `Keypoints.swift` | Keypoint index construction |
| `projection.py` | `Projection.swift` | Identical cubic model |
| `solve.py` | `Solver.swift` | Uses OpenCV solvePnP via bridge |
| `optimise/` | `PowellOptimizer.swift` + `Objective.swift` | Custom Powell instead of SciPy |
| `dewarp.py` | `Remapper.swift` | Same decimation grid approach |
| `normalisation.py` | `Normalisation.swift` | Identical math |
| `mask.py` | `OpenCVWrapper.mm` | Mask pipeline moved into ObjC++ bridge |
| `options/` | `DewarpConfig.swift` | Static constants instead of msgspec Struct |

## Key Differences

### 1. Image I/O and EXIF Orientation

**Python**: `cv2.imread()` in OpenCV 4.x automatically applies EXIF rotation. Images are NumPy arrays with shape `(height, width, channels)`.

**Swift**: `UIImage` preserves EXIF orientation metadata separately from pixel data. The `OpenCVWrapper.mm` bridge normalizes orientation before converting to `cv::Mat` by rendering through a `CGBitmapContext` that respects `UIImage.imageOrientation`. Without this fix, portrait photos would be processed sideways.

### 2. OpenCV Access via ObjC++ Bridge

**Python**: Direct `cv2` calls. NumPy arrays pass through seamlessly.

**Swift**: All OpenCV calls go through an ObjC++ wrapper (`OpenCVWrapper.mm`). Data types are bridged:
- `UIImage` <-> `cv::Mat` (via `CGBitmapContext`)
- `[NSValue]` (wrapping `CGPoint`) <-> `std::vector<cv::Point>`
- `[NSNumber]` <-> `std::vector<float/double>`
- `NSDictionary` <-> structured return values (moments, SVD results, solvePnP output)

This adds marshalling overhead but preserves correctness. All internal computation uses `Double` (float64) matching Python's precision.

### 3. Optimizer: Powell vs L-BFGS-B

**Python**: Uses `scipy.optimize.minimize(method='L-BFGS-B')` with JAX autodiff providing analytical gradients. Falls back to Powell if JAX is unavailable.

**Swift**: Uses a custom implementation of Powell's conjugate direction method with Brent's 1D line search, ported from SciPy's `_optimize.py`. No gradient computation.

Implications:
- Swift needs more function evaluations to converge (~600K max vs ~300 typical for L-BFGS-B)
- Powell may find different local minima, leading to slightly different (but visually acceptable) results
- Processing time: 2-17 seconds per image (Swift/Powell) vs 0.3-0.5 seconds (Python/JAX)

### 4. Mask Computation Location

**Python**: `mask.py` contains the `Mask.calculate()` method, which orchestrates grayscale conversion, adaptive thresholding, morphological operations, and page mask application.

**Swift**: This entire sequence is implemented in `OpenCVWrapper.mm` as `computeDetectionMask:pagemask:isText:adaptiveWinsz:`. Moving it to ObjC++ avoids multiple Swift-to-ObjC-to-C++ round trips and keeps all the `cv::Mat` operations in one place.

### 5. Configuration

**Python**: `Config` is a `msgspec.Struct` with a global `cfg` singleton. CLI args update the singleton.

**Swift**: `DewarpConfig` is a struct with static constants. No CLI, no runtime configuration. All parameters are compile-time constants matching Python's defaults:
- `screenMaxW/H`: 1280x700
- `pageMarginX/Y`: 50x20
- `adaptiveWinsz`: 55
- `outputZoom`: 1.0
- `remapDecimate`: 16
- `optMaxIter`: 600,000
- `focalLength`: 1.2

### 6. Error Handling

**Python**: Prints warnings and continues (e.g., "only got N spans" logged but processing continues with what's available).

**Swift**: Returns `Result<UIImage, DewarpError>` with typed errors:
- `.noContoursDetected`
- `.insufficientSpans(count:)`
- `.solvePnPFailed`
- `.invalidPageDimensions`

### 7. Numerical Precision

Both implementations use float64 (Double) for the parameter vector and optimizer state. The OpenCV bridge uses float32 for image coordinate maps (matching Python's behavior in `cv2.remap`).

Known precision differences:
- `norm2pix` truncates toward zero (matching Python's `int()`) rather than floor/round
- Powell's direction set update uses the SciPy heuristic for dropping the direction of maximum decrease
- Brent's line search tolerance matches SciPy's default: `1.48e-8`

## Output Quality

Tested on 20 recipe book photos:
- Pixel match with Python: 81-96% (average ~89%)
- All images produce visually correct, flat, readable output
- Dimension match: within 2-10% of Python
- The gap is primarily due to Powell vs L-BFGS-B finding slightly different optima
