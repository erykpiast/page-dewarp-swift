# Task Breakdown: Make OpenCV a Peer Dependency

Generated: 2026-04-08
Source: specs/feat-opencv-peer-dependency.md

## Overview

Replace the last three `calib3d` OpenCV functions with pure-Swift equivalents, remove dead bridge code, and add a CocoaPods podspec so consuming apps (e.g., `fasola`) can share their own `opencv-rne` instead of pulling in a vendored binary.

## Phase 1: Pure-Swift solvePnP

### Task 1.1: Capture golden solvePnP values

**Description**: Before replacing anything, capture the current OpenCV solvePnP output as golden reference data for cross-validation tests.
**Size**: Small
**Priority**: High
**Dependencies**: None
**Can run parallel with**: Nothing (must be first)

**Implementation Steps**:
1. Add a test or script that calls `OpenCVWrapper.solvePnP()` with the golden corners from `Tests/SolverTests.swift` and prints the exact rvec/tvec values.
2. Also capture `projectXY()` output for 3 representative inputs with the golden pvec — these become golden values for Task 2.3.
3. Save these values as constants in a new golden test data file or as inline constants in the test files.

**Golden data to capture** (from `Tests/SolverTests.swift`):
```swift
// Inputs already available in SolverTests:
let goldenCorners: [[Double]] = [
    [-0.5962007285432904, -0.9444430488014438],
    [ 0.6076691006680317, -0.9371052476847103],
    [ 0.5962007285432904,  0.9444430488014438],
    [-0.6076691006680317,  0.9371052476847103],
]
// Expected outputs already captured:
let expectedRvec: [Double] = [-0.0, 0.0, 0.0060951]
let expectedTvec: [Double] = [-0.59620073, -0.94444305, 1.20000005]
```

Also capture `projectXY()` output for:
```swift
let xyCoords = [[0.1, 0.2], [-0.3, 0.15], [0.0, 0.0]]
// with goldenPvec from ProjectionTests
```

**Acceptance Criteria**:
- [ ] Golden rvec/tvec values captured from current OpenCV solvePnP
- [ ] Golden projectXY output captured for 3 representative inputs
- [ ] Values saved as test constants for later cross-validation

---

### Task 1.2: Implement `solvePnPPlanar()` and `rotationMatrixToRvec()`

**Description**: Create pure-Swift DLT homography + decomposition to replace `cv::solvePnP` for the planar case (all Z=0). Includes the inverse Rodrigues function.
**Size**: Large
**Priority**: High
**Dependencies**: None (can develop in parallel with 1.1)
**Can run parallel with**: Task 1.1

**New file**: `Sources/PageDewarp/SolvePnPDLT.swift`

**Algorithm — DLT for planar homography**:

Since all object points have z=0, the projection equation simplifies to a 3×3 homography:
```
s · [u, v, 1]^T = K · [r1 | r2 | t] · [X, Y, 1]^T
```

Each of the 4 correspondences gives 2 equations in the 8×9 DLT system:
```
[ X  Y  1  0  0  0  -uX  -uY  -u ] · h = 0
[ 0  0  0  X  Y  1  -vX  -vY  -v ] · h = 0
```

The homography h is the right singular vector corresponding to the smallest singular value.

**Implementation of `solvePnPPlanar`**:
```swift
import Foundation
import Accelerate

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
        A[row0*9 + 0] = X;  A[row0*9 + 1] = Y;  A[row0*9 + 2] = 1
        A[row0*9 + 6] = -u*X; A[row0*9 + 7] = -u*Y; A[row0*9 + 8] = -u
        A[row1*9 + 3] = X;  A[row1*9 + 4] = Y;  A[row1*9 + 5] = 1
        A[row1*9 + 6] = -v*X; A[row1*9 + 7] = -v*Y; A[row1*9 + 8] = -v
    }

    // SVD of A (8×9) using LAPACK dgesdd_
    // Extract last column of V^T → h (the null vector)
    // Reshape h into 3×3 homography H

    // Decompose: M = K^{-1} · H = [r1 | r2 | t] (up to scale)
    // Normalize: scale = ||col1(M)||
    // r1 = M[:,0] / scale, r2 = M[:,1] / scale, r3 = r1 × r2, t = M[:,2] / scale
    // Project R onto SO(3) via SVD: R_approx = U·S·V^T → R = U·V^T
    // Convert R to rvec via rotationMatrixToRvec()

    return (rvec: rvec, tvec: tvec)
}
```

**Implementation of `rotationMatrixToRvec`** (handles small angle, general, AND near-pi cases):
```swift
func rotationMatrixToRvec(_ R: [Double]) -> [Double] {
    let trace = R[0] + R[4] + R[8]
    let cosAngle = min(1.0, max(-1.0, (trace - 1.0) / 2.0))
    let angle = acos(cosAngle)

    // Small angle: rvec ≈ [R[7]-R[5], R[2]-R[6], R[3]-R[1]] / 2
    if angle < 1e-10 {
        return [(R[7]-R[5])/2, (R[2]-R[6])/2, (R[3]-R[1])/2]
    }

    // Near-pi: sin(θ) ≈ 0, so skew-symmetric extraction is unstable.
    // Extract axis from symmetric part S = (R + I) / 2.
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

    // General case: axis from skew-symmetric part
    let s = 1.0 / (2.0 * sin(angle))
    let axis = [(R[7] - R[5]) * s, (R[2] - R[6]) * s, (R[3] - R[1]) * s]
    return axis.map { $0 * angle }
}
```

**Key LAPACK details**:
- Use `dgesdd_` for both the 8×9 DLT SVD and the 3×3 orthogonal projection SVD
- LAPACK uses column-major storage — transpose when building A
- The `Accelerate` framework is already linked in `Package.swift`

**Acceptance Criteria**:
- [ ] `solvePnPPlanar()` compiles and links via `import Accelerate`
- [ ] Returns nil for degenerate inputs (collinear points, < 4 points)
- [ ] `rotationMatrixToRvec()` round-trips with `rodriguesRotationOnly()` to < 1e-12 error
- [ ] `rotationMatrixToRvec()` handles identity (returns [0,0,0])
- [ ] `rotationMatrixToRvec()` handles near-pi rotations without NaN

---

### Task 1.3: Wire solvePnPPlanar into Solver.swift and validate

**Description**: Replace `OpenCVWrapper.solvePnP()` in `Solver.swift:getDefaultParams()` with the new pure-Swift `solvePnPPlanar()`. Remove NSNumber conversions.
**Size**: Small
**Priority**: High
**Dependencies**: Task 1.1, Task 1.2
**Can run parallel with**: Nothing

**File to modify**: `Sources/PageDewarp/Solver.swift`

**Before** (current code, lines 38-70):
```swift
let objectPoints: [NSNumber] = [
    0, 0, 0,
    NSNumber(value: pageWidth), 0, 0,
    NSNumber(value: pageWidth), NSNumber(value: pageHeight), 0,
    0, NSNumber(value: pageHeight), 0,
]
let imagePoints: [NSNumber] = corners.flatMap { pt in
    [NSNumber(value: pt[0]), NSNumber(value: pt[1])]
}
let kFlat = cameraMatrix().flatMap { $0 }.map { NSNumber(value: $0) }
let distCoeffs: [NSNumber] = [0, 0, 0, 0, 0]

let pnpResult = OpenCVWrapper.solvePnP(
    withObjectPoints: objectPoints,
    imagePoints: imagePoints,
    cameraMatrix: kFlat,
    distCoeffs: distCoeffs
)
guard let success = pnpResult["success"] as? NSNumber, success.boolValue else {
    return .failure(.solvePnPFailed)
}
let rvecArr = (pnpResult["rvec"] as! [NSNumber]).map { $0.doubleValue }
let tvecArr = (pnpResult["tvec"] as! [NSNumber]).map { $0.doubleValue }
```

**After**:
```swift
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

Also remove `import OpenCVBridge` if Solver.swift no longer calls any OpenCV bridge methods.

**Acceptance Criteria**:
- [ ] `Solver.swift` no longer imports or calls `OpenCVWrapper`
- [ ] `getDefaultParams()` returns success with the golden corner inputs
- [ ] rvec/tvec from DLT produce projected points within 1e-6 of OpenCV golden values
- [ ] Full pipeline golden image regression passes (PSNR > 40 dB)
- [ ] All existing `SolverTests` pass (they test `getDefaultParams()` outputs)

---

## Phase 2: Remove Dead calib3d Code

### Task 2.1: Delete `projectXY()` from Projection.swift

**Description**: Remove the OpenCV-backed `projectXY()` function. It has zero production callers — `projectXYPure()` and `projectXYBulk()` fully replace it.
**Size**: Small
**Priority**: High
**Dependencies**: Task 1.3 (must be complete so no code path needs projectXY)
**Can run parallel with**: Task 2.2

**File**: `Sources/PageDewarp/Projection.swift`

Delete lines 170–228 (the `projectXY()` function including its doc comment). The function starts at:
```swift
/// Projects normalized (x, y) coordinates through a cubic warp surface model into image space.
///
/// Builds a cubic polynomial z(x) = ((a·x + b)·x + c)·x from pvec's cubic coefficients,
/// then calls OpenCV projectPoints to map (x, y, z) 3D points into 2D image coordinates.
func projectXY(xyCoords: [[Double]], pvec: [Double]) -> [[Double]] {
```

After deletion, also check if `import OpenCVBridge` can be removed from `Projection.swift` (it can — `projectXYPure` and `projectXYBulk` don't use it).

**Acceptance Criteria**:
- [ ] `projectXY()` function deleted from `Projection.swift`
- [ ] `import OpenCVBridge` removed from `Projection.swift` if no other bridge calls remain
- [ ] Project compiles without errors
- [ ] All tests that previously called `projectXY()` are updated (see Task 2.3)

---

### Task 2.2: Remove calib3d bridge methods from OpenCVWrapper

**Description**: Remove the 3 calib3d-dependent methods from `OpenCVWrapper.h` and `OpenCVWrapper.mm`, plus the `#import <opencv2/calib3d.hpp>`.
**Size**: Small
**Priority**: High
**Dependencies**: Task 1.3
**Can run parallel with**: Task 2.1

**Files**: `Sources/OpenCVBridge/include/OpenCVWrapper.h`, `Sources/OpenCVBridge/OpenCVWrapper.mm`

**Remove from OpenCVWrapper.h** (declarations):
1. Lines 55-65: `solvePnPWithObjectPoints:imagePoints:cameraMatrix:distCoeffs:` and its doc comment
2. Lines 67-77: `projectPointsWith3DPoints:rvec:tvec:cameraMatrix:distCoeffs:` and its doc comment
3. Lines 79-82: `rodriguesFromVector:` and its doc comment

**Remove from OpenCVWrapper.mm** (implementations):
1. The `solvePnPWithObjectPoints:` method implementation
2. The `projectPointsWith3DPoints:` method implementation
3. The `rodriguesFromVector:` method implementation
4. The import: `#import <opencv2/calib3d.hpp>`

**Acceptance Criteria**:
- [ ] `calib3d.hpp` import removed from `OpenCVWrapper.mm`
- [ ] 3 method declarations removed from `OpenCVWrapper.h`
- [ ] 3 method implementations removed from `OpenCVWrapper.mm`
- [ ] Project compiles (note: existing XCFramework still has calib3d, it just won't be used)
- [ ] All remaining bridge methods (`findContours`, `moments`, `remap`, etc.) still work

---

### Task 2.3: Migrate test files

**Description**: Update the 4 test files that reference removed functions. Convert cross-validation tests to golden-value tests.
**Size**: Medium
**Priority**: High
**Dependencies**: Task 1.1 (golden values), Task 2.1, Task 2.2
**Can run parallel with**: Nothing

**Test files to migrate**:

1. **`Tests/ProjectionTests.swift`** (6 calls to `projectXY()`):
   - Replace all `projectXY()` calls with `projectXYPure()`
   - The tests verify clamping, golden outputs, etc. — these should still pass since `projectXYPure` produces identical results
   - Update the file header comment from "Tests for projectXY()" to "Tests for projectXYPure()"

2. **`Tests/PerfProjectionTests.swift`** (7 calls to `projectXY()`):
   - Remove the `projectXY` benchmark and the cross-comparison benchmark (`projectXY` vs `projectXYPure`)
   - Keep `projectXYPure` and `projectXYBulk` benchmarks
   - Remove any `import OpenCVBridge` if no longer needed

3. **`Tests/PureProjectionTests.swift`** (uses `OpenCVWrapper.rodrigues(fromVector:)` and `projectPointsWith3DPoints`):
   - Replace `OpenCVWrapper.rodrigues(fromVector:)` cross-validation with golden-value assertions using constants captured in Task 1.1
   - Replace `OpenCVWrapper.projectPointsWith3DPoints` cross-validation similarly
   - Keep the pure-Swift Rodrigues tests (Jacobian, small angle, etc.) — they don't use OpenCV

4. **`Tests/OpenCVBridgeTests.swift`** (tests `rodriguesFromVector` and `projectPointsWith3DPoints`):
   - Delete `testRodriguesZeroVectorIsIdentity()` and `testProjectPointsOriginStaysAtPrincipalPoint()`
   - Keep all remaining bridge tests (contour detection, moments, remap, etc.)

**Acceptance Criteria**:
- [ ] All 4 test files updated
- [ ] No test file references `projectXY()`, `OpenCVWrapper.rodrigues`, `OpenCVWrapper.projectPointsWith3DPoints`, or `OpenCVWrapper.solvePnP`
- [ ] All tests pass
- [ ] No `import OpenCVBridge` in test files that no longer need it

---

## Phase 3: CocoaPods Podspec

### Task 3.1: Create PageDewarp.podspec

**Description**: Add a CocoaPods podspec that declares `opencv-rne ~> 4.11` as a peer dependency, using subspecs for correct mixed-language compilation.
**Size**: Medium
**Priority**: High
**Dependencies**: Task 2.1, Task 2.2 (calib3d must be removed first so the lib compiles with only core+imgproc)
**Can run parallel with**: Task 2.3

**New file**: `PageDewarp.podspec` (project root)

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

  s.dependency 'opencv-rne', '~> 4.11'

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

**Key design decisions**:
- Subspecs separate Swift, ObjC++, and C compilation (each needs different compiler settings)
- `opencv-rne` dependency is declared at top level, not per-subspec
- `static_framework = true` for compatibility with React Native/Expo
- No manual `HEADER_SEARCH_PATHS` — CocoaPods resolves them from declared dependencies

**Acceptance Criteria**:
- [ ] `PageDewarp.podspec` exists at project root
- [ ] `pod lib lint PageDewarp.podspec` passes (or passes with `--allow-warnings`)
- [ ] Subspecs correctly separate Swift/ObjC++/C sources
- [ ] `CLBFGSB` module.modulemap and header search paths work under CocoaPods
- [ ] `OpenCVBridge` finds OpenCV headers via the `opencv-rne` dependency

---

### Task 3.2: Update documentation

**Description**: Update README and architecture docs to reflect the new CocoaPods distribution and calib3d removal.
**Size**: Small
**Priority**: Medium
**Dependencies**: Task 3.1
**Can run parallel with**: Nothing

**Files to update**:

1. **`README.md`** — Add CocoaPods installation section:
   ```ruby
   # Podfile
   pod 'PageDewarp', '~> 2.0'
   ```
   Note that `opencv-rne` is pulled in automatically.

2. **`docs/architecture.md`** — Update the OpenCV bridge diagram to show only `core` + `imgproc` (remove calib3d). Add note about the new `SolvePnPDLT.swift` module.

3. **`docs/API.md`** — No changes needed (public API unchanged).

**Acceptance Criteria**:
- [ ] README has CocoaPods integration instructions
- [ ] Architecture docs reflect calib3d removal
- [ ] Minimum OpenCV requirements documented (`core` + `imgproc`)

---

## Dependency Graph

```
Phase 1:
  1.1 (golden values) ──┐
  1.2 (solvePnPPlanar)──┴── 1.3 (wire into Solver)

Phase 2:
  1.3 ──┬── 2.1 (delete projectXY) ──┐
        └── 2.2 (remove bridge)  ────┤
  1.1 ──────────────────────────────┴── 2.3 (migrate tests)

Phase 3:
  2.1 + 2.2 ── 3.1 (podspec)
  3.1 ──────── 3.2 (docs)
```

## Parallel Execution Opportunities

- **Task 1.1** and **Task 1.2** can run in parallel
- **Task 2.1** and **Task 2.2** can run in parallel
- **Task 3.1** can run in parallel with **Task 2.3**

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| DLT produces different rvec/tvec than OpenCV solvePnP | High (expected) | Low — optimizer converges to same result | Golden image PSNR test validates end-to-end equivalence |
| LAPACK dgesdd_ API differences from expected | Low | Medium | Accelerate is well-documented; existing project already uses vDSP |
| Podspec fails `pod lib lint` | Medium | Medium | Iterate on subspec configuration; test with local pod install first |
| `module.modulemap` conflicts between SPM and CocoaPods | Medium | Medium | SPM's `exclude: ["module.modulemap"]` already handles this; verify CocoaPods path |
