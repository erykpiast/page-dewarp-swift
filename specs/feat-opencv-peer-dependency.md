# Feature: Make OpenCV a Peer Dependency (Remove Vendored Binary)

- **Status**: Draft
- **Authors**: Claude Code, 2026-04-08
- **Type**: Feature (architecture / distribution)

---

## Overview

Restructure page-dewarp-swift so it does not vendor its own OpenCV binary. Instead, the consuming project provides OpenCV via CocoaPods (`opencv-rne`), and page-dewarp-swift links against it. This requires (1) replacing the last three `calib3d` functions with pure-Swift equivalents so only `core` + `imgproc` are needed, and (2) adding a CocoaPods podspec that declares `opencv-rne` as a peer dependency.

## Background / Problem Statement

page-dewarp-swift currently vendors a custom-built `opencv2.xcframework` (~50–100 MB) as a binary SPM target hosted on GitHub Releases. This creates two problems for consuming apps:

1. **Duplicate OpenCV**: Apps like `fasola` that already use OpenCV (e.g., `opencv-rne ~> 4.11` via CocoaPods) end up with two copies of OpenCV — one vendored by page-dewarp-swift (4.10.0) and one from the host project. This causes linker conflicts (duplicate symbols) or silent ABI mismatches.

2. **Version lock-in**: The vendored binary is pinned to OpenCV 4.10.0. If the host project needs a different version (4.11, 5.x, or a custom build with specific modules), there is no way to override it without forking page-dewarp-swift.

3. **Binary size**: Consuming apps pay the full cost of the vendored XCFramework even if they already link OpenCV.

### Why calib3d removal is the key enabler

The vendored XCFramework currently includes `calib3d` because three bridge functions use it:
- `cv::solvePnP` — called once in `Solver.swift` for initial pose estimation
- `cv::projectPoints` — called in `Projection.swift:projectXY()` but already fully replaced by `projectXYPure()` and `projectXYBulk()`
- `cv::Rodrigues` — already fully replaced in `PureProjection.swift`

`calib3d` pulls in transitive dependencies (`features2d → flann`) that roughly double the binary size. Removing it means the library only needs `core` + `imgproc` — the two modules present in every OpenCV build, including the most minimal configurations. This makes the "use whatever OpenCV the host provides" approach viable, because any OpenCV build will satisfy the requirement.

## Goals

- Replace `cv::solvePnP` with a pure-Swift DLT + SVD implementation
- Remove `projectXY()` (OpenCV-backed) — it is fully superseded by `projectXYPure()` and `projectXYBulk()`
- Remove `cv::Rodrigues` bridge call — already replaced by `rodrigues()` in `PureProjection.swift`
- Remove the `calib3d.hpp` import from `OpenCVWrapper.mm`
- Add a CocoaPods podspec declaring `opencv-rne ~> 4.11` as a peer dependency
- Provide clear CocoaPods integration documentation
- Maintain visually identical output (PSNR > 40 dB vs current golden images)

## Non-Goals

- Replacing `core` or `imgproc` OpenCV functions with pure Swift (contour detection, remap, adaptive threshold, morphological ops, SVD, PCA — these stay)
- Dropping the XcodeGen/SPM build path (it should continue to work for local dev)
- Supporting macOS, tvOS, watchOS, or visionOS
- Replacing OpenCV entirely — `imgproc` algorithms (findContours, remap, adaptiveThreshold) have no practical pure-Swift alternative at comparable quality and performance

## Technical Dependencies

### OpenCV modules still required after this change

| Module | Functions used | Why |
|--------|---------------|-----|
| `core` | `SVDecomp`, `PCACompute`, `Mat`, `Scalar`, conversions | Linear algebra for blob orientation and PCA |
| `imgproc` | `findContours`, `moments`, `boundingRect`, `convexHull`, `remap`, `resize`, `adaptiveThreshold`, `dilate`, `erode`, `cvtColor`, `drawContours`, `getStructuringElement`, `reduce`, `minMaxLoc`, `bitwise_and`, `rectangle` | All image processing operations |

### OpenCV modules removed

| Module | Functions replaced | Replacement |
|--------|-------------------|-------------|
| `calib3d` | `solvePnP` | Pure-Swift DLT (new, this spec) |
| `calib3d` | `projectPoints` | `projectXYPure()`, `projectXYBulk()` (already exist) |
| `calib3d` | `Rodrigues` | `rodrigues()` in `PureProjection.swift` (already exists) |

### Accelerate framework

The `Accelerate` framework (`vDSP`, `LAPACK`) is already linked. The pure-Swift solvePnP will use `LAPACK` for SVD, which is the same underlying implementation OpenCV uses on Apple platforms.

## Detailed Design

### Phase 1: Replace `solvePnP` with pure-Swift DLT

#### The problem

`cv::solvePnP` is called exactly once per pipeline run in `Solver.swift:getDefaultParams()` with:
- 4 object points (3D corners of a flat page, z=0)
- 4 image points (2D corners in normalized coordinates)
- Camera matrix K = diag(1.2, 1.2, 1) with zero principal point
- Zero distortion coefficients

It returns `rvec` (3×1 rotation vector) and `tvec` (3×1 translation vector).

#### DLT approach for the planar case

Since all object points have z=0 (planar target), this is a homography estimation problem. The standard approach:

1. **Compute the 3×3 homography H** from 4 coplanar point correspondences using the Direct Linear Transform (DLT).
2. **Decompose H into R and t** using the known camera matrix K.

##### Step 1: DLT homography

Given 4 correspondences (X_i, Y_i) ↔ (u_i, v_i) where the object points are planar (Z=0), the projection equation simplifies to:

```
s · [u, v, 1]^T = K · [r1 | r2 | t] · [X, Y, 1]^T
```

where `H = K · [r1 | r2 | t]` is a 3×3 homography. Each correspondence gives 2 equations:

```
[ X  Y  1  0  0  0  -uX  -uY  -u ] · h = 0
[ 0  0  0  X  Y  1  -vX  -vY  -v ] · h = 0
```

With 4 points this yields an 8×9 system. The homography h is the null vector (right singular vector corresponding to the smallest singular value) from SVD.

##### Step 2: Decompose H → R, t

```
M = K^{-1} · H = [r1 | r2 | t] (up to scale)
```

Normalize so that `||col1(M)|| = 1`. Then:
- `r1 = M[:,0]` (already unit length after normalization)
- `r2 = M[:,1]` (normalize to unit length)
- `r3 = r1 × r2` (cross product ensures orthogonality)
- `t = M[:,2]`

The rotation matrix `R = [r1 | r2 | r3]` may not be exactly orthogonal due to noise, so we project it onto SO(3) via SVD: `R = U · V^T` where `R_approx = U · S · V^T`.

Finally, convert R to rvec via the inverse Rodrigues formula (angle-axis extraction from rotation matrix).

##### Implementation

New file: `Sources/PageDewarp/SolvePnPDLT.swift`

```swift
import Foundation
import Accelerate

/// Estimate rotation and translation from 4 coplanar 3D↔2D correspondences.
///
/// Pure-Swift replacement for cv::solvePnP, specialized for the planar case (all Z=0).
/// Uses DLT homography estimation + decomposition into R,t.
///
/// - Parameters:
///   - objectPoints: Flat [X0,Y0,Z0, X1,Y1,Z1, ...] — must have Z=0 for all points.
///   - imagePoints: Flat [u0,v0, u1,v1, ...]
///   - cameraMatrix: Flat 3×3 row-major [fx,0,cx, 0,fy,cy, 0,0,1]
/// - Returns: (success, rvec, tvec) or nil on failure.
func solvePnPPlanar(
    objectPoints: [Double],
    imagePoints: [Double],
    cameraMatrix: [Double]
) -> (rvec: [Double], tvec: [Double])? {
    let n = objectPoints.count / 3
    guard n >= 4, imagePoints.count == n * 2 else { return nil }

    // Build 2n×9 DLT matrix A
    var A = [Double](repeating: 0, count: 2 * n * 9)
    for i in 0..<n {
        let X = objectPoints[i*3], Y = objectPoints[i*3+1]
        let u = imagePoints[i*2], v = imagePoints[i*2+1]
        let row0 = 2*i, row1 = 2*i + 1
        // Row 0: [X Y 1  0 0 0  -uX -uY -u]
        A[row0*9 + 0] = X;  A[row0*9 + 1] = Y;  A[row0*9 + 2] = 1
        A[row0*9 + 6] = -u*X; A[row0*9 + 7] = -u*Y; A[row0*9 + 8] = -u
        // Row 1: [0 0 0  X Y 1  -vX -vY -v]
        A[row1*9 + 3] = X;  A[row1*9 + 4] = Y;  A[row1*9 + 5] = 1
        A[row1*9 + 6] = -v*X; A[row1*9 + 7] = -v*Y; A[row1*9 + 8] = -v
    }

    // SVD of A to find null vector → homography h
    // Use LAPACK dgesdd (already available via Accelerate)
    var h = [Double](repeating: 0, count: 9)
    // ... (LAPACK SVD call, extract last column of V^T)

    // Reshape h → 3×3 homography H
    // M = K^{-1} · H
    // Decompose M into [r1 | r2 | t], enforce orthogonality, extract rvec
    // ...

    return (rvec: rvec, tvec: tvec)
}
```

The actual implementation will use `dgesdd_` from LAPACK (via `import Accelerate`) for the 8×9 SVD and the 3×3 orthogonal projection SVD. This is the same LAPACK backend that OpenCV uses on Apple platforms.

#### Inverse Rodrigues (rotation matrix → rotation vector)

Required to convert the decomposed rotation matrix R back to the 3-element rvec format used by the parameter vector. Implementation:

```swift
/// Convert a 3×3 rotation matrix to a rotation vector (inverse of rodrigues()).
///
/// Handles three regimes: small angle (θ ≈ 0), general case, and near-pi (θ ≈ π).
/// The near-pi case extracts the axis from the symmetric part (R + I) to avoid
/// the numerical instability of dividing by sin(θ) → 0.
///
/// - Parameter R: 9-element flat rotation matrix (row-major).
/// - Returns: 3-element rotation vector [r1, r2, r3].
func rotationMatrixToRvec(_ R: [Double]) -> [Double] {
    let trace = R[0] + R[4] + R[8]
    let cosAngle = min(1.0, max(-1.0, (trace - 1.0) / 2.0))
    let angle = acos(cosAngle)

    // Small angle: rvec ≈ [R[7]-R[5], R[2]-R[6], R[3]-R[1]] / 2
    if angle < 1e-10 {
        return [(R[7]-R[5])/2, (R[2]-R[6])/2, (R[3]-R[1])/2]
    }

    // Near-pi: sin(θ) ≈ 0, so skew-symmetric extraction is unstable.
    // Extract axis from the symmetric part S = (R + I) / 2 = k·k^T + cos(θ)·I ≈ k·k^T.
    // The axis is the column of S with the largest diagonal entry, normalized.
    if angle > .pi - 1e-6 {
        let s00 = (R[0] + 1) / 2, s11 = (R[4] + 1) / 2, s22 = (R[8] + 1) / 2
        var axis: [Double]
        if s00 >= s11 && s00 >= s22 {
            let k0 = sqrt(s00)
            axis = [k0, (R[1]+R[3])/(4*k0), (R[2]+R[6])/(4*k0)]
        } else if s11 >= s22 {
            let k1 = sqrt(s11)
            axis = [(R[1]+R[3])/(4*k1), k1, (R[5]+R[7])/(4*k1)]
        } else {
            let k2 = sqrt(s22)
            axis = [(R[2]+R[6])/(4*k2), (R[5]+R[7])/(4*k2), k2]
        }
        let norm = sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2])
        return axis.map { $0 / norm * angle }
    }

    // General case: axis from skew-symmetric part of R
    let s = 1.0 / (2.0 * sin(angle))
    let axis = [
        (R[7] - R[5]) * s,
        (R[2] - R[6]) * s,
        (R[3] - R[1]) * s,
    ]
    return axis.map { $0 * angle }
}
```

### Phase 2: Remove dead OpenCV bridge code

#### Remove `projectXY()` from `Projection.swift`

Delete `projectXY()` (lines 181–228). Verify no callers remain — currently:
- `projectXYPure()` is used by `projectKeypoints()` in `Keypoints.swift` and `getPageDims()` in `DewarpPipeline.swift`
- `projectXYBulk()` is used by `RemappedImage` in `Remapper.swift`
- `projectXY()` has **zero callers** in production code (may have test callers for cross-validation)

#### Remove bridge methods from `OpenCVWrapper`

Remove from `OpenCVWrapper.h` and `OpenCVWrapper.mm`:
1. `solvePnPWithObjectPoints:imagePoints:cameraMatrix:distCoeffs:` — replaced by `solvePnPPlanar()`
2. `projectPointsWith3DPoints:rvec:tvec:cameraMatrix:distCoeffs:` — replaced by `projectXYPure()` / `projectXYBulk()`
3. `rodriguesFromVector:` — replaced by `rodrigues()` in `PureProjection.swift`

#### Remove `calib3d` import from `OpenCVWrapper.mm`

```diff
 #import <opencv2/core.hpp>
 #import <opencv2/imgproc.hpp>
-#import <opencv2/calib3d.hpp>
```

#### Update `Solver.swift`

Replace the `OpenCVWrapper.solvePnP()` call with the new `solvePnPPlanar()`. Also remove the `[NSNumber]` construction — the pure-Swift function works with `[Double]` directly:

```swift
// Before:
let objectPoints: [NSNumber] = [0, 0, 0, NSNumber(value: pageWidth), 0, 0, ...]
let imagePoints: [NSNumber] = corners.flatMap { pt in [NSNumber(value: pt[0]), ...] }
let kFlat = cameraMatrix().flatMap { $0 }.map { NSNumber(value: $0) }
let pnpResult = OpenCVWrapper.solvePnP(withObjectPoints: objectPoints, ...)

// After:
let objectPoints: [Double] = [0, 0, 0, pageWidth, 0, 0,
                               pageWidth, pageHeight, 0, 0, pageHeight, 0]
let imagePoints: [Double] = corners.flatMap { $0 }
let kFlat: [Double] = cameraMatrix().flatMap { $0 }

guard let pnpResult = solvePnPPlanar(
    objectPoints: objectPoints,
    imagePoints: imagePoints,
    cameraMatrix: kFlat
) else {
    return .failure(.solvePnPFailed)
}
let rvecArr = pnpResult.rvec
let tvecArr = pnpResult.tvec
```

### Phase 3: Add CocoaPods podspec with `opencv-rne` peer dependency

The consuming app (`fasola`) is an Expo/React Native project that uses CocoaPods exclusively. The distribution path is CocoaPods, not SPM.

`Package.swift` remains unchanged for the library's own development and testing (it keeps the vendored `binaryTarget`). The podspec is the consumer-facing interface.

Create `PageDewarp.podspec`:

```ruby
Pod::Spec.new do |s|
  s.name         = 'PageDewarp'
  s.version      = '2.0.0'
  s.summary      = 'Page dewarping for document images'
  s.homepage     = 'https://github.com/erykpiast/page-dewarp-swift'
  s.license      = { type: 'MIT' }
  s.authors      = { 'Eryk Napierala' => '...' }
  s.source       = { git: 'https://github.com/erykpiast/page-dewarp-swift.git', tag: s.version }

  s.ios.deployment_target = '16.0'
  s.swift_version = '5.9'
  s.static_framework = true
  s.libraries = 'c++'
  s.frameworks = 'UIKit', 'Accelerate'

  # Peer dependency: links against the same OpenCV the host project provides.
  # Requires core + imgproc modules (present in all opencv-rne builds).
  s.dependency 'opencv-rne', '~> 4.11'

  # Subspecs handle mixed-language compilation (Swift, ObjC++, C).
  s.subspec 'CLBFGSB' do |c|
    c.source_files = 'Sources/CLBFGSB/**/*.{c,h}'
    c.public_header_files = 'Sources/CLBFGSB/include/*.h'
    c.header_dir = 'CLBFGSB'
  end

  s.subspec 'OpenCVBridge' do |b|
    b.source_files = 'Sources/OpenCVBridge/**/*.{h,m,mm}'
    b.public_header_files = 'Sources/OpenCVBridge/include/*.h'
    b.pod_target_xcconfig = {
      'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17',
    }
  end

  s.subspec 'Core' do |core|
    core.source_files = 'Sources/PageDewarp/**/*.swift'
    core.dependency 'PageDewarp/OpenCVBridge'
    core.dependency 'PageDewarp/CLBFGSB'
  end

  s.default_subspecs = 'Core'
end
```

Note: CocoaPods automatically adds header search paths for declared dependencies (`opencv-rne`), so no manual `HEADER_SEARCH_PATHS` is needed. The exact paths will be validated during `pod lib lint`.

When the consuming app's Podfile already has `pod 'opencv-rne'`, CocoaPods resolves to a single shared copy. No duplicate symbols, no version conflicts.

### File changes summary

| File | Action | Description |
|------|--------|-------------|
| `Sources/PageDewarp/SolvePnPDLT.swift` | **New** | Pure-Swift DLT solvePnP for planar targets |
| `Sources/PageDewarp/Solver.swift` | **Modify** | Replace `OpenCVWrapper.solvePnP()` with `solvePnPPlanar()`, remove NSNumber conversions |
| `Sources/PageDewarp/Projection.swift` | **Modify** | Remove `projectXY()` function (lines 170–228 including doc comment) |
| `Sources/OpenCVBridge/include/OpenCVWrapper.h` | **Modify** | Remove 3 calib3d method declarations |
| `Sources/OpenCVBridge/OpenCVWrapper.mm` | **Modify** | Remove 3 calib3d method implementations + `#import <opencv2/calib3d.hpp>` |
| `Package.swift` | **No change** | Keeps vendored `binaryTarget` for local dev/testing |
| `PageDewarp.podspec` | **New** | CocoaPods podspec with peer dependency on `opencv-rne` |
| `Tests/ProjectionTests.swift` | **Modify** | Convert `projectXY()` calls to `projectXYPure()` with golden values |
| `Tests/PerfProjectionTests.swift` | **Modify** | Remove `projectXY` benchmarks, keep `projectXYPure`/`projectXYBulk` |
| `Tests/PureProjectionTests.swift` | **Modify** | Convert OpenCV cross-validation to golden-value assertions |
| `Tests/OpenCVBridgeTests.swift` | **Modify** | Delete calib3d test cases, keep imgproc/core tests |

### Code that does NOT change

These files call OpenCV only for `imgproc`/`core` functions and require no modification:

| File | OpenCV functions used |
|------|----------------------|
| `ContourDetector.swift` | `findContours`, `moments`, `boundingRect`, `svDecomp`, `maxColumnThickness` |
| `SpanAssembler.swift` | `columnMeans`, `pcaCompute`, `convexHull` |
| `Remapper.swift` | `resizeFloatMapData`, `remapImageData`, `remapColorImageData`, `adaptiveThresholdImage` |
| `DewarpPipeline.swift` | `resize`, `createPageMask`, `computeDetectionMask` |

## User Experience

### For library consumers (primary audience)

**Before**: No CocoaPods distribution. Consumers must integrate via SPM or manual embedding, pulling in a vendored OpenCV binary that may conflict with their own.

**After**: Add `pod 'PageDewarp'` to the Podfile. CocoaPods resolves the `opencv-rne` dependency automatically. If the host app already has `pod 'opencv-rne'`, a single shared copy is used. No duplicate symbols, no version conflicts.

```ruby
# fasola Podfile
pod 'opencv-rne', '~> 4.11'   # already present
pod 'PageDewarp', '~> 2.0'    # new — shares the same OpenCV
```

### API surface

The public API (`DewarpPipeline.process(image:method:output:)`) does not change. This is a purely internal refactoring.

## Testing Strategy

### 1. Cross-validation: solvePnPPlanar vs OpenCV solvePnP

Before removing the OpenCV bridge calls, validate that `solvePnPPlanar()` produces equivalent results:

```swift
/// Purpose: Verify the pure-Swift DLT solvePnP produces rvec/tvec values
/// that, when used in the pipeline, yield the same projected points as
/// the OpenCV solvePnP result. This is the critical correctness gate.
func testSolvePnPPlanarMatchesOpenCV() {
    // Use the same 4-corner inputs from a real pipeline run (golden data)
    let objectPoints: [Double] = [0, 0, 0, 0.5, 0, 0, 0.5, 0.3, 0, 0, 0.3, 0]
    let imagePoints: [Double] = [/* golden values from a captured test case */]
    let K: [Double] = [1.2, 0, 0, 0, 1.2, 0, 0, 0, 1]

    let result = solvePnPPlanar(
        objectPoints: objectPoints,
        imagePoints: imagePoints,
        cameraMatrix: K
    )!

    // Project the 4 object points using both rvec/tvec results
    let projPure = projectXYPure(xyCoords: corners, pvec: buildPvec(result))
    let projOpenCV = projectXYPure(xyCoords: corners, pvec: buildPvec(opencvResult))

    // The projected points must match within tolerance.
    // Tolerance: 1e-6 in normalized coordinates (sub-pixel in image space).
    for i in 0..<4 {
        XCTAssertEqual(projPure[i][0], projOpenCV[i][0], accuracy: 1e-6)
        XCTAssertEqual(projPure[i][1], projOpenCV[i][1], accuracy: 1e-6)
    }
}
```

### 2. End-to-end golden image regression

```swift
/// Purpose: Verify that the full pipeline produces visually identical output
/// after replacing solvePnP. The DLT may produce slightly different rvec/tvec
/// than OpenCV's iterative solvePnP, but the optimizer converges to the same
/// final result. PSNR > 40 dB confirms visual equivalence.
func testGoldenImageUnchanged() {
    let input = loadTestImage("IMG_1369")
    let result = DewarpPipeline.process(image: input)
    let output = try! result.get()

    let golden = loadGoldenImage("IMG_1369_dewarped")
    // PSNR > 40 dB means visually identical
    XCTAssertGreaterThan(psnr(output, golden), 40.0)
}
```

### 3. DLT edge cases

```swift
/// Purpose: Verify DLT handles degenerate inputs gracefully (collinear points).
func testSolvePnPPlanarRejectsCollinearPoints() {
    let collinear: [Double] = [0,0,0, 1,0,0, 2,0,0, 3,0,0]
    let imagePoints: [Double] = [0,0, 1,0, 2,0, 3,0]
    let K: [Double] = [1,0,0, 0,1,0, 0,0,1]
    XCTAssertNil(solvePnPPlanar(objectPoints: collinear, imagePoints: imagePoints, cameraMatrix: K))
}

/// Purpose: Verify inverse Rodrigues (R → rvec) round-trips correctly.
func testRotationMatrixToRvecRoundTrip() {
    let rvecOriginal = [0.3, -0.2, 0.1]
    let R = rodriguesRotationOnly(rvecOriginal)
    let rvecRecovered = rotationMatrixToRvec(R)
    for i in 0..<3 {
        XCTAssertEqual(rvecRecovered[i], rvecOriginal[i], accuracy: 1e-12)
    }
}

/// Purpose: Verify identity rotation produces zero rvec.
func testRotationMatrixToRvecIdentity() {
    let I = [1.0, 0, 0, 0, 1, 0, 0, 0, 1.0]
    let rvec = rotationMatrixToRvec(I)
    for r in rvec {
        XCTAssertEqual(r, 0.0, accuracy: 1e-14)
    }
}
```

### 4. Verify projectXY removal

```swift
/// Purpose: Confirm that projectXYPure matches the (removed) projectXY for
/// representative inputs. Run this BEFORE deleting projectXY to capture golden values.
func testProjectXYPureMatchesGolden() {
    let xyCoords = [[0.1, 0.2], [-0.3, 0.15], [0.0, 0.0]]
    let pvec = loadGoldenPvec("IMG_1369_optimized_params")
    let result = projectXYPure(xyCoords: xyCoords, pvec: pvec)
    let golden = loadGoldenProjection("IMG_1369_projectXY_golden")
    for i in 0..<3 {
        XCTAssertEqual(result[i][0], golden[i][0], accuracy: 1e-10)
        XCTAssertEqual(result[i][1], golden[i][1], accuracy: 1e-10)
    }
}
```

### 5. Test file migration

The following test files reference functions being removed and must be updated:

| Test file | Impact | Migration |
|-----------|--------|-----------|
| `Tests/ProjectionTests.swift` | 6 calls to `projectXY()` | Convert to golden-value tests using `projectXYPure()`. Capture golden projected values from current `projectXY()` before deleting it. |
| `Tests/PerfProjectionTests.swift` | 7 calls to `projectXY()` (benchmarks) | Remove `projectXY` benchmarks. Keep `projectXYPure` and `projectXYBulk` benchmarks. The cross-comparison benchmark (`projectXY` vs `projectXYPure`) is no longer meaningful — delete it. |
| `Tests/PureProjectionTests.swift` | Uses `OpenCVWrapper.rodrigues(fromVector:)` and `projectPointsWith3DPoints` for cross-validation | Convert cross-validation assertions to golden-value assertions. Capture reference values from current OpenCV output before removing bridge methods. |
| `Tests/OpenCVBridgeTests.swift` | Tests `rodriguesFromVector` and `projectPointsWith3DPoints` directly | Delete the calib3d-specific test cases. Keep tests for remaining bridge methods (`findContours`, `moments`, `remap`, etc.). |

## Performance Considerations

| Operation | Before | After | Impact |
|-----------|--------|-------|--------|
| `solvePnP` (1 call/run) | ~0.1ms (OpenCV C++) | ~0.2ms (Swift + LAPACK SVD) | Negligible — called once per image, not in any hot loop |
| `projectXY` → `projectXYPure` | Already using pure Swift in hot loop | No change | None |
| OpenCV binary size | ~100MB (core+imgproc+calib3d+features2d+flann) | ~50MB (core+imgproc only) | ~50% reduction for standalone usage |
| Linker time (consumer) | May fail with duplicate symbols | Clean link | Eliminates integration blocker |

The solvePnP replacement is not performance-sensitive. It runs once per image, taking <1ms. The pipeline's bottleneck is the optimizer loop (hundreds of iterations), which already uses pure-Swift `projectXYPure()` and `projectXYBulk()`.

## Security Considerations

No new security surface. The DLT implementation operates on numeric arrays only — no file I/O, no network, no string parsing.

## Documentation

- Update `README.md` with CocoaPods integration instructions
- Update `docs/architecture.md` to reflect the removal of calib3d and the new DLT module
- Document the minimum OpenCV module requirements (`core` + `imgproc`)

## Implementation Phases

### Phase 1: Pure-Swift solvePnP (core change)

1. Implement `solvePnPPlanar()` in new file `Sources/PageDewarp/SolvePnPDLT.swift`
2. Implement `rotationMatrixToRvec()` (inverse Rodrigues)
3. Add cross-validation tests comparing DLT output to OpenCV solvePnP output (capture golden rvec/tvec from current OpenCV implementation first)
4. Wire `solvePnPPlanar()` into `Solver.swift`, replacing `OpenCVWrapper.solvePnP()` directly
5. Run golden image regression to verify visually identical output (PSNR > 40 dB)

### Phase 2: Remove dead calib3d code

1. Delete `projectXY()` from `Projection.swift`
2. Remove `solvePnPWithObjectPoints:`, `projectPointsWith3DPoints:`, `rodriguesFromVector:` from `OpenCVWrapper.h` and `OpenCVWrapper.mm`
3. Remove `#import <opencv2/calib3d.hpp>` from `OpenCVWrapper.mm`
4. Update tests that cross-validated against OpenCV functions (convert to golden-value tests)
5. Rebuild the `opencv2.xcframework` without `calib3d` for local dev (optional — existing binary still works, just carries unused code)

### Phase 3: Add CocoaPods podspec

1. Create `PageDewarp.podspec` with `s.dependency 'opencv-rne', '~> 4.11'`
2. Validate podspec with `pod lib lint`
3. Test integration in a sample CocoaPods project that also uses `opencv-rne`
4. Update README with CocoaPods integration instructions
5. Tag version 2.0.0

## Resolved Questions

1. **Distribution mechanism**: CocoaPods only (consuming app is an Expo project using CocoaPods exclusively). No SPM wrapper package needed. `Package.swift` stays as-is for local dev.

2. **Minimum OpenCV version**: Pin to `opencv-rne ~> 4.11`. Can relax later if someone requests it.

3. **XCFramework hosting**: Not needed. CocoaPods consumers get OpenCV from `opencv-rne`. The existing hosted XCFramework stays for local SPM-based dev/testing but is not part of the consumer-facing distribution.

4. **2×2 SVD in `blobMeanAndTangent`**: Keep using `OpenCVWrapper.svDecomp()` for now. Can optimize later — it still requires `core`, which we're keeping anyway.

5. **Major version bump**: Yes, tag as 2.0.0.

## References

- `Sources/PageDewarp/PureProjection.swift` — existing pure-Swift Rodrigues and projectPoints implementations
- `Sources/PageDewarp/Projection.swift` — `projectXYPure()` and `projectXYBulk()` that replace `projectXY()`
- `Sources/PageDewarp/Solver.swift` — the sole caller of `cv::solvePnP`
- `Sources/OpenCVBridge/OpenCVWrapper.mm` — the ObjC++ bridge implementation
- `specs/feat-swift-package-manager.md` — prior spec that established the current SPM + vendored binary approach
- OpenCV `cv::solvePnP` documentation: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d
- DLT for homography estimation: Hartley & Zisserman, "Multiple View Geometry", Chapter 4
- LAPACK `dgesdd` documentation (used via `import Accelerate` on Apple platforms)
