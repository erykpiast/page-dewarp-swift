# Architecture

## Pipeline Overview

```
Input UIImage
       |
       v
  +-----------+
  |  Resize   |  DewarpPipeline.swift (resizeToScreen)
  +-----------+
       |
       v
  +-----------+
  | Page Mask |  DewarpPipeline.swift (makePageExtents)
  +-----------+
       |
       v
  +-----------+
  |  Contour  |  ContourDetector.swift + OpenCVWrapper (computeDetectionMask)
  | Detection |
  +-----------+
       |
       v
  +-----------+
  |   Span    |  SpanAssembler.swift
  | Assembly  |
  +-----------+
       |
       v
  +-----------+
  | Keypoint  |  Keypoints.swift (sampleSpans, keypointsFromSamples)
  | Sampling  |
  +-----------+
       |
       v
  +-----------+
  |  Initial  |  Solver.swift (solvePnP via OpenCVWrapper)
  |  Params   |
  +-----------+
       |
       v
  +-----------+
  |  Powell   |  PowellOptimizer.swift + Objective.swift
  | Optimize  |
  +-----------+
       |
       v
  +-----------+
  |   Remap   |  Remapper.swift + OpenCVWrapper (remap, adaptiveThreshold)
  | Threshold |
  +-----------+
       |
       v
  Output UIImage
```

## Source Files

### Core Pipeline

| File | Python equivalent | Responsibility |
|------|------------------|----------------|
| `DewarpPipeline.swift` | `image.py` | Orchestrates all pipeline stages. Entry point: `process(image:)` |
| `DewarpConfig.swift` | `options/core.py` | All tunable parameters (margins, thresholds, optimizer settings) |
| `AlgorithmCore.swift` | (various) | Shared helpers: `vecNorm`, `linspace`, `roundNearestMultiple` |

### Detection & Analysis

| File | Python equivalent | Responsibility |
|------|------------------|----------------|
| `ContourDetector.swift` | `contours.py` | Text contour detection, filtering by size/aspect/thickness |
| `SpanAssembler.swift` | `spans.py` | Links contours into horizontal spans via edge scoring |
| `Keypoints.swift` | `keypoints.py` + `spans.py` | Samples points along spans, computes PCA orientation and corners |

### 3D Modeling & Optimization

| File | Python equivalent | Responsibility |
|------|------------------|----------------|
| `Projection.swift` | `projection.py` | Cubic polynomial surface model, 3D-to-2D projection via OpenCV |
| `Solver.swift` | `solve.py` | Initial parameter estimation via `solvePnP` |
| `Objective.swift` | `optimise/_base.py` | Builds the least-squares objective function for optimization |
| `PowellOptimizer.swift` | SciPy's `_optimize.py` | Powell's conjugate direction method with Brent's line search |
| `CameraMatrix.swift` | `projection.py` | Camera intrinsic matrix construction |
| `Normalisation.swift` | `normalisation.py` | Pixel-to-normalized coordinate transforms |

### Output

| File | Python equivalent | Responsibility |
|------|------------------|----------------|
| `Remapper.swift` | `dewarp.py` | Builds decimated coordinate maps, remaps image, applies adaptive threshold |

### OpenCV Bridge

| File | Responsibility |
|------|----------------|
| `OpenCVWrapper.h` | Public ObjC header declaring all bridge methods |
| `OpenCVWrapper.mm` | ObjC++ implementation wrapping C++ OpenCV calls |
| `OpenCVBridge.h` | Umbrella header for the bridge module |
| `module.modulemap` | Clang module map exposing ObjC++ to Swift |

The bridge is necessary because OpenCV has no official Swift bindings. All OpenCV calls flow through this ObjC++ layer:

```
Swift code  -->  OpenCVWrapper (ObjC++)  -->  cv:: C++ API
   ^                                              |
   |                                              v
UIImage  <--  CGBitmapContext  <--  cv::Mat
```

Key bridge methods:
- `findContoursInGrayImage:` -- contour detection
- `computeDetectionMask:pagemask:isText:adaptiveWinsz:` -- full mask pipeline
- `solvePnPWithObjectPoints:imagePoints:cameraMatrix:distCoeffs:` -- pose estimation
- `projectPointsWith3DPoints:rvec:tvec:cameraMatrix:distCoeffs:` -- 3D projection
- `remapImage:mapX:mapY:width:height:` -- image warping
- `adaptiveThresholdImage:maxValue:blockSize:C:` -- binarization

## Parameter Vector Layout

The optimizer works on a single flat parameter vector:

```
Index  0-2:   rvec     (Rodrigues rotation vector)
Index  3-5:   tvec     (translation vector)
Index  6-7:   alpha, beta  (cubic polynomial coefficients, clamped to [-0.5, 0.5])
Index  8+:    ycoords  (one per span), then all xcoords concatenated
```

Total length depends on number of spans and keypoints, typically 300-400 parameters.

## Cubic Surface Model

The page is modeled as a cubic polynomial surface in 3D:

```
z(x) = ((a*x + b)*x + c)*x

where:
  alpha = pvec[6], beta = pvec[7]
  a = alpha + beta
  b = -2*alpha - beta
  c = alpha
```

This maps each (x, y) page coordinate to a 3D point (x, y, z), which is then projected to 2D via OpenCV's `projectPoints` using the camera matrix and pose (rvec, tvec).
