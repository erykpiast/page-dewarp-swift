# Feature: Analytical Gradient for L-BFGS-B Optimizer

- **Status**: Draft
- **Authors**: Claude Code, 2026-04-03
- **Type**: Feature (performance)

---

## Overview

Replace the finite-difference numerical gradient in `LBFGSBOptimizer.swift` with an analytical gradient computed via the chain rule. This eliminates ~592 extra objective evaluations per L-BFGS-B iteration, reducing total function evaluations from ~97K to ~100 and achieving ~1000x speedup in the optimization step.

## Background / Problem Statement

The L-BFGS-B optimizer (just integrated) uses **forward finite differences** to approximate the gradient:

```
grad[i] ≈ (f(x + h·e_i) - f(x)) / h    for each of N parameters
```

With N ≈ 592 parameters, each gradient computation costs 592 extra objective evaluations. The optimizer typically needs ~100 gradient evaluations, giving **~60K total evaluations** — better than Powell's 258K but still far from optimal.

The Python reference uses **JAX automatic differentiation**, which computes the exact gradient in a single reverse pass at roughly the same cost as one function evaluation. This is why Python processes images in 0.3–0.5s while Swift takes 47s on the same image.

### Current evaluation results (from this session)

| Metric | L-BFGS-B + finite diff | Powell | Python (JAX) |
|--------|----------------------|--------|--------------|
| Function evals | 96,659 | 257,999 | ~300 |
| Time (simulator) | 47s | 72s | 0.3–0.5s |
| Final loss | 0.00426 | 0.00443 | 0.00487 |

### Why analytical gradients are feasible

The objective function is a composition of well-understood, differentiable operations:

```
pvec → index into (rvec, tvec, cubic, ycoords, xcoords)
     → build 3D points (x, y, z(x)) via cubic polynomial
     → Rodrigues rotation + translation
     → perspective division
     → camera matrix multiplication
     → sum of squared errors against target points
```

Every step has a known, closed-form derivative. No operation is opaque — even `cv::projectPoints` is just Rodrigues + pinhole projection, which we can reimplement in ~40 lines of Swift.

### Why not use OpenCV's built-in Jacobian?

`cv::projectPoints` can output a Jacobian matrix, and our current code already calls it (the `jacobian` parameter in `OpenCVWrapper.mm:344`). However:

1. OpenCV's Jacobian is w.r.t. `(rvec, tvec, f, cx, cy, distCoeffs)` — only 14 columns. Our parameter vector has ~592 entries including per-keypoint coordinates and cubic coefficients.
2. We'd still need to chain-rule through the cubic polynomial and keypoint indexing.
3. Each call still crosses the ObjC++ bridge. With an analytical gradient, we eliminate bridge crossings entirely during gradient computation.

Reimplementing `projectPoints` in pure Swift is simpler and faster than trying to extract and chain OpenCV's partial Jacobian.

## Goals

- Implement pure-Swift `projectPoints` (Rodrigues + pinhole camera model) replacing the OpenCV bridge call during optimization
- Implement analytical Jacobian of the full objective function
- Pass the analytical gradient to `lbfgsbMinimize` instead of `finiteDifferenceGradient`
- Reduce total evaluations from ~97K to ~200 (100 iterations × 2 evaluations for function + gradient)
- Match Python's processing speed within 2–5x (target: <2s per image on simulator)
- Maintain identical optimization results (same minimum found, same output image)

## Non-Goals

- Replacing OpenCV for `solvePnP` (initial parameter estimation — called once, not in the optimization loop)
- Replacing OpenCV for image processing (thresholding, contours, remapping)
- General-purpose automatic differentiation framework
- GPU acceleration

## Technical Dependencies

- No new dependencies. This removes the OpenCV dependency from the hot loop.
- Existing `LBFGSBOptimizer.swift` is modified to accept an optional gradient function.
- Existing `Projection.swift` gets a pure-Swift companion.

## Detailed Design

### The Math

The full derivative chain, from objective to parameters:

#### 1. Objective function

```
f(pvec) = Σᵢ ||proj(Pᵢ) - dstᵢ||² + shearCost · pvec[0]²
```

where `proj(P)` projects a 3D point to 2D image coordinates.

Gradient of the squared error term w.r.t. projected point:
```
∂f/∂projᵢ = 2 · (projᵢ - dstᵢ)    (2D vector per point)
```

#### 2. Pinhole projection (what `cv::projectPoints` does)

Given a 3D point `P = (X, Y, Z)` in object space:

**Step 2a: Rodrigues rotation**

Convert rotation vector `r = (r₁, r₂, r₃)` to rotation matrix `R`:
```
θ = ||r||
k = r / θ                    (unit axis)
K = [0, -k₃, k₂; k₃, 0, -k₁; -k₂, k₁, 0]   (skew-symmetric)
R = I + sin(θ)·K + (1 - cos(θ))·K²
```

**Step 2b: Transform to camera space**
```
P_cam = R · P + t           (3D vector)
```

**Step 2c: Perspective division + camera matrix**
```
x' = P_cam.x / P_cam.z
y' = P_cam.y / P_cam.z
u = f · x'                  (f = focal length = 1.2)
v = f · y'
```

No distortion (distCoeffs = 0), no principal point offset (cx = cy = 0).

#### 3. Rodrigues Jacobian (dR/dr)

The 9×3 Jacobian of the rotation matrix elements w.r.t. the rotation vector. This is the most complex piece of math. Reference: OpenCV's `cvRodrigues2` in `modules/calib3d/src/calibration.cpp` (lines ~550–650).

Each element `R_ij` of the rotation matrix depends on `r` through `θ`, `k`, `sin(θ)`, `cos(θ)`:

```
R = cos(θ)·I + (1 - cos(θ))·k·kᵀ + sin(θ)·[k]×
```

The derivative of each term w.r.t. `r_m` (where m ∈ {1,2,3}):

```
∂θ/∂r_m = r_m / θ = k_m

∂k_i/∂r_m = (δ_im - k_i·k_m) / θ     (where δ is Kronecker delta)

∂R_ij/∂r_m = -sin(θ)·k_m·δ_ij                                    // from cos(θ)·I
            + sin(θ)·k_m·(k_i·k_j)                                 // from (1-cos(θ))·k·kᵀ (θ part)
            + (1-cos(θ))·(∂k_i/∂r_m·k_j + k_i·∂k_j/∂r_m)         // from (1-cos(θ))·k·kᵀ (k part)
            + cos(θ)·k_m·[k]×_ij                                   // from sin(θ)·[k]× (θ part)
            + sin(θ)·∂[k]×_ij/∂r_m                                 // from sin(θ)·[k]× (k part)
```

where `∂[k]×_ij/∂r_m` is the derivative of the skew-symmetric matrix entries w.r.t. `r_m`, computed via `∂k/∂r`.

**Small-angle case (θ < 1e-10):** `R ≈ I + [r]×`, so `∂R/∂r` is simply the derivative of the skew-symmetric mapping:
```
∂R/∂r₁ = [0,0,0; 0,0,-1; 0,1,0]
∂R/∂r₂ = [0,0,1; 0,0,0; -1,0,0]
∂R/∂r₃ = [0,-1,0; 1,0,0; 0,0,0]
```

#### 4. Perspective division Jacobian

For `(u, v) = f · (cx/cz, cy/cz)`:

```
∂u/∂cx = f/cz       ∂u/∂cy = 0          ∂u/∂cz = -f·cx/cz²
∂v/∂cx = 0           ∂v/∂cy = f/cz       ∂v/∂cz = -f·cy/cz²
```

This 2×3 matrix `J_persp` chains with `∂(cx,cy,cz)/∂(anything)` to give `∂(u,v)/∂(anything)`.

#### 5. Cubic polynomial

From `Projection.swift:19-29`:
```
α = clamp(pvec[6], -0.5, 0.5)
β = clamp(pvec[7], -0.5, 0.5)
a = α + β,  b = -2α - β,  c = α

z(x) = ((a·x + b)·x + c)·x = (α+β)x³ - (2α+β)x² + αx
```

Derivatives:
```
∂z/∂x = 3(α+β)x² - 2(2α+β)x + α
∂z/∂α = x³ - 2x² + x       (zero when |pvec[6]| ≥ 0.5, matching JAX clamp behavior)
∂z/∂β = x³ - x²             (zero when |pvec[7]| ≥ 0.5)
```

#### 6. Keypoint indexing

From `Keypoints.swift:51-57`:
```
xyCoords[k] = [pvec[keypointIndex[k][0]], pvec[keypointIndex[k][1]]]
```

This is a simple gather — the Jacobian is a sparse selection matrix.

#### 7. Gradient accumulation — complete chain

For each keypoint `k`, compute `P_cam = R · (x_k, y_k, z(x_k))ᵀ + t`, then project.

Let `J_persp` be the 2×3 perspective Jacobian and `e_k = 2·(proj_k - dst_k)` be the 2D error gradient.

**rvec** (indices 0–2): Every keypoint contributes.
```swift
// dP_cam/dr is a 3×3 matrix computed from dR/dr · P
// For each component m in 0..<3:
//   dP_cam_x/dr_m = dR[0,m]*X + dR[1,m]*Y + dR[2,m]*Z   (row 0 of dR/dr_m · P)
//   dP_cam_y/dr_m = dR[3,m]*X + dR[4,m]*Y + dR[5,m]*Z
//   dP_cam_z/dr_m = dR[6,m]*X + dR[7,m]*Y + dR[8,m]*Z
// Then: grad[m] += e_k · J_persp · dP_cam/dr_m
for m in 0..<3 {
    let dcx = dR_dr[0*3+m]*X + dR_dr[1*3+m]*Y + dR_dr[2*3+m]*Z
    let dcy = dR_dr[3*3+m]*X + dR_dr[4*3+m]*Y + dR_dr[5*3+m]*Z
    let dcz = dR_dr[6*3+m]*X + dR_dr[7*3+m]*Y + dR_dr[8*3+m]*Z
    let du_dr = J00*dcx + J02*dcz    // J01=0
    let dv_dr = J11*dcy + J12*dcz    // J10=0
    grad[m] += eu*du_dr + ev*dv_dr
}
```

**tvec** (indices 3–5): Every keypoint contributes. `dP_cam/dt = I`, so:
```swift
grad[3] += eu * J00                  // du/dtx = f/cz
grad[4] += ev * J11                  // dv/dty = f/cz
grad[5] += eu * J02 + ev * J12      // d(u,v)/dtz = (-f·cx/cz², -f·cy/cz²)
```

**cubic α, β** (indices 6–7): Every keypoint contributes via `z(x)`.
```swift
let dz_dalpha = x*x*x - 2*x*x + x   // zero if |pvec[6]| >= 0.5
let dz_dbeta  = x*x*x - x*x          // zero if |pvec[7]| >= 0.5
// dP_cam/dz = R's third column: (R02, R12, R22)
let du_dz = J00*R02 + J02*R22        // J01=0, R column 2
let dv_dz = J11*R12 + J12*R22
grad[6] += (eu*du_dz + ev*dv_dz) * dz_dalpha
grad[7] += (eu*du_dz + ev*dv_dz) * dz_dbeta
```

**ycoords** (indices 8..8+nspans): `dP_cam/dy = R's second column`.
```swift
let du_dy = J00*R01 + J02*R21
let dv_dy = J11*R11 + J12*R21
grad[yIdx] += eu*du_dy + ev*dv_dy
```

**xcoords** (indices 8+nspans..end): Has two paths — direct and via z(x).
```swift
let dz_dx = 3*(alpha+beta)*x*x - 2*(2*alpha+beta)*x + alpha
// dP_cam/dx = R's first column + R's third column · dz/dx
let du_dx = J00*(R00 + R02*dz_dx) + J02*(R20 + R22*dz_dx)
let dv_dx = J11*(R10 + R12*dz_dx) + J12*(R20 + R22*dz_dx)
grad[xIdx] += eu*du_dx + ev*dv_dx
```

### New File: `PureProjection.swift`

Pure-Swift reimplementation of `cv::projectPoints` for the optimization loop. Uses `[Double]` arrays for Jacobians (not tuples).

```swift
/// Rodrigues rotation: convert rotation vector to 3x3 matrix and its 9×3 Jacobian.
/// Matches OpenCV's cv::Rodrigues exactly.
///
/// - Parameter rvec: 3-element rotation vector.
/// - Returns: (R as [Double] row-major 9 elements, dR_dr as [Double] 27 elements in row-major 9×3)
func rodrigues(_ rvec: [Double]) -> (R: [Double], dR_dr: [Double]) {
    // ... implementation using formulas from section 3 above
}

/// Project N 3D points through the pinhole camera model.
/// Returns projected 2D points and per-point Jacobians.
///
/// - Parameters:
///   - points3D: Flat array [x0,y0,z0, x1,y1,z1, ...] of N points.
///   - rvec: 3-element rotation vector.
///   - tvec: 3-element translation vector.
///   - focalLength: Scalar focal length (1.2).
/// - Returns:
///   - projected: Flat [u0,v0, u1,v1, ...] of N projected 2D points.
///   - dProj_dPoint: Flat [N × 6] — each point's 2×3 Jacobian w.r.t. (X,Y,Z), row-major.
///   - dProj_dRvec: Flat [N × 6] — each point's 2×3 Jacobian w.r.t. rvec.
///   - dProj_dTvec: Flat [N × 6] — each point's 2×3 Jacobian w.r.t. tvec.
func projectAndDifferentiate(
    points3D: [Double],
    rvec: [Double],
    tvec: [Double],
    focalLength: Double
) -> (projected: [Double], dProj_dPoint: [Double], dProj_dRvec: [Double], dProj_dTvec: [Double])
```

### New File: `AnalyticalGradient.swift`

Assembles the full gradient from the chain rule, using the accumulation patterns from section 7.

```swift
/// Compute the objective value AND its analytical gradient in a single pass.
///
/// This replaces the pattern of calling makeObjective() + finiteDifferenceGradient().
/// The gradient is exact (not approximated), computed via chain rule through:
///   pvec → keypoint indexing → cubic z(x) → Rodrigues + pinhole → squared error
///
/// - Returns: (f: objective value, grad: gradient vector of length pvec.count)
func objectiveAndGradient(
    pvec: [Double],
    dstpoints: [[Double]],
    keypointIndex: [[Int]],
    shearCost: Double,
    focalLength: Double
) -> (f: Double, grad: [Double])
```

### Changes to `LBFGSBOptimizer.swift`

Add overload that accepts a combined objective+gradient function:

```swift
/// L-BFGS-B with caller-provided gradient (no finite differences).
func lbfgsbMinimize(
    objectiveAndGradient: ([Double]) -> (f: Double, grad: [Double]),
    x0: [Double],
    maxIter: Int = 600_000,
    maxCor: Int = 100,
    factr: Double = 1e7,
    pgtol: Double = 1e-5
) -> OptimizeResult {
    // Same setulb() reverse-communication loop as existing overload.
    // On "FG" task, instead of:
    //   f = objective(x); g = finiteDifferenceGradient(...)
    // Do:
    //   let (fVal, gVal) = objectiveAndGradient(x)
    //   f = fVal; g = gVal
    // nfev increments by 1 per call (not N+1).
}
```

### Changes to `DewarpPipeline.swift`

```swift
// Before:
let optResult = lbfgsbMinimize(objective: objective, x0: initialParams)

// After:
let gradObjective: ([Double]) -> (f: Double, grad: [Double]) = { pvec in
    objectiveAndGradient(
        pvec: pvec, dstpoints: dstpoints, keypointIndex: keypointIndex,
        shearCost: DewarpConfig.shearCost, focalLength: DewarpConfig.focalLength
    )
}
let optResult = lbfgsbMinimize(objectiveAndGradient: gradObjective, x0: initialParams)
```

### File Organization

```
Sources/Core/
  PureProjection.swift        (NEW — ~150 lines: Rodrigues + pinhole + Jacobians)
  AnalyticalGradient.swift    (NEW — ~120 lines: chain rule assembly)
  LBFGSBOptimizer.swift       (MODIFIED — add objectiveAndGradient overload)
  DewarpPipeline.swift        (MODIFIED — use new overload)
  Projection.swift             (UNCHANGED — still used by getPageDims and Remapper)
  Objective.swift              (UNCHANGED — still used by getPageDims)
```

The existing `Projection.swift` (with OpenCV bridge) is kept for use by `Remapper` and the page-dimension optimization, which are not in the hot loop.

## User Experience

No API changes. Same `DewarpPipeline.process(image:)`. Users observe significantly faster processing.

## Testing Strategy

### Unit Tests (`PureProjectionTests.swift`)

**1. Rodrigues rotation accuracy**
```swift
// Compare rodrigues([0.1, 0.2, 0.3]) output against OpenCVWrapper.rodriguesFromVector
// Purpose: validate pure-Swift Rodrigues matches OpenCV to machine precision
// Tolerance: 1e-10
```

**2. Rodrigues small-angle case**
```swift
// Compare rodrigues([1e-12, 0, 0]) against OpenCV
// Purpose: validate the θ≈0 branch doesn't produce NaN or diverge
// Tolerance: 1e-10
```

**3. Rodrigues Jacobian accuracy**
```swift
// For rvec = [0.1, 0.2, 0.3], compare analytical dR/dr against finite-difference:
//   dR_dr_fd[i][m] = (rodrigues(r + h·e_m).R[i] - rodrigues(r).R[i]) / h
// Purpose: catch sign errors or missing terms in the 27-entry Jacobian
// Tolerance: 1e-6
```

**4. Pure projectPoints vs OpenCV**
```swift
// Project N=100 random 3D points with rvec/tvec from golden file
// Compare pure-Swift output against OpenCVWrapper.projectPointsWith3DPoints
// Purpose: validate the pure-Swift projection matches OpenCV exactly
// Tolerance: 1e-8
```

**5. Projection Jacobian accuracy**
```swift
// For the same N=100 points, compare analytical dProj/dPoint, dProj/dRvec, dProj/dTvec
// against finite-difference Jacobians of the pure-Swift projection
// Purpose: catch chain-rule errors in the projection derivatives
// Tolerance: 1e-5
```

### Unit Tests (`AnalyticalGradientTests.swift`)

**6. Cubic polynomial derivatives**
```swift
// For x=0.3, α=0.15, β=0.1:
//   Verify dz/dx = 3(α+β)x² - 2(2α+β)x + α against finite diff of z(x)
//   Verify dz/dα = x³ - 2x² + x against finite diff of z(α)
//   Verify dz/dβ = x³ - x² against finite diff of z(β)
// Purpose: catch algebraic errors in the cubic derivative formulas
// Tolerance: 1e-6
```

**7. Cubic clamp boundary**
```swift
// For α = 0.5 (at clamp boundary):
//   Verify dz/dα = 0 (gradient is zero when clamped)
// Purpose: validate clamp derivative matches JAX behavior
```

**8. Full gradient vs finite differences (CRITICAL TEST)**
```swift
// Load golden file initial_params.json as pvec
// Build keypointIndex and dstpoints from golden data
// Compute objectiveAndGradient(pvec, ...)
// Compute finiteDifferenceGradient(makeObjective(...), pvec, f0)
// Compare element-by-element
// Purpose: end-to-end validation that the analytical gradient is correct
// If this test passes, the entire chain rule is correct
// Tolerance: 1e-4 (limited by finite-difference accuracy)
```

**9. Gradient at known point is nonzero**
```swift
// Use golden file initial_params.json (not at optimum)
// Compute gradient, verify max(|grad|) > 0.01
// Purpose: smoke test — the gradient should be meaningful, not zero
```

### Integration Tests — Real-Case Validation

**10. Side-by-side optimizer comparison on golden file**

This is the key validation test. Run the full optimizer with both gradient methods on the same golden file input and compare all metrics:

```swift
func testAnalyticalVsFiniteDiffOnGoldenFile() throws {
    // Setup: load golden file, build objective
    let image = try GoldenFileLoader.loadImage("boston_cooking_a_input.jpg")
    // ... (reconstruct dstpoints, keypointIndex, initialParams as in EvalComparisonTests)

    // Run L-BFGS-B with FINITE-DIFFERENCE gradient
    let fdResult = lbfgsbMinimize(objective: objective, x0: initialParams)

    // Run L-BFGS-B with ANALYTICAL gradient
    let agResult = lbfgsbMinimize(objectiveAndGradient: gradObjective, x0: initialParams)

    // Load Python reference
    let golden: GoldenParams = try GoldenFileLoader.loadJSON("optimized_params")

    // --- Convergence comparison ---
    // Both must converge
    XCTAssertTrue(fdResult.converged)
    XCTAssertTrue(agResult.converged)

    // Analytical must use dramatically fewer evaluations
    // Finite-diff: ~97K evals (observed). Analytical: should be <500.
    XCTAssertLessThan(agResult.nfev, 1000, "Analytical should need <1000 evals")
    print("Function evals — FD: \(fdResult.nfev), Analytical: \(agResult.nfev)")

    // --- Objective value comparison ---
    // Both should find similar minima (within 5% relative)
    let relDiff = abs(fdResult.fun - agResult.fun) / max(abs(fdResult.fun), 1e-10)
    XCTAssertLessThan(relDiff, 0.05, "Final loss should be within 5%")
    print("Final loss — FD: \(fdResult.fun), Analytical: \(agResult.fun), Python: \(golden.final_loss ?? -1)")

    // --- Parameter comparison ---
    // rvec, tvec, cubic should be close between methods
    for i in 0..<8 {
        let diff = abs(fdResult.x[i] - agResult.x[i])
        XCTAssertLessThan(diff, 0.05, "Param[\(i)] should match within 0.05")
    }
    print("rvec — FD: \(fdResult.x[0..<3]), AG: \(agResult.x[0..<3]), Python: \(golden.rvec)")
    print("tvec — FD: \(fdResult.x[3..<6]), AG: \(agResult.x[3..<6]), Python: \(golden.tvec)")
    print("cubic — FD: \(fdResult.x[6..<8]), AG: \(agResult.x[6..<8]), Python: \(golden.cubic)")
}
```

**Expected output** (to be filled in after implementation):
```
Function evals — FD: ~97000, Analytical: ~200
Final loss — FD: 0.00426, Analytical: ~0.00426, Python: 0.00487
rvec — FD: [-0.078, 0.073, 0.012], AG: [similar], Python: [-0.057, 0.071, 0.011]
tvec — FD: [-0.606, -0.961, 1.221], AG: [similar], Python: [-0.605, -0.958, 1.218]
cubic — FD: [0.187, 0.136], AG: [similar], Python: [0.194, 0.137]
```

**11. Wall-clock timing comparison**
```swift
func testTimingComparison() throws {
    // Same setup as test 10

    // Time the finite-difference optimizer
    let fdStart = CFAbsoluteTimeGetCurrent()
    let fdResult = lbfgsbMinimize(objective: objective, x0: initialParams)
    let fdTime = CFAbsoluteTimeGetCurrent() - fdStart

    // Time the analytical optimizer
    let agStart = CFAbsoluteTimeGetCurrent()
    let agResult = lbfgsbMinimize(objectiveAndGradient: gradObjective, x0: initialParams)
    let agTime = CFAbsoluteTimeGetCurrent() - agStart

    let speedup = fdTime / agTime

    print("--- Timing Results ---")
    print("Finite-diff: \(String(format: "%.2f", fdTime))s (\(fdResult.nfev) evals)")
    print("Analytical:  \(String(format: "%.2f", agTime))s (\(agResult.nfev) evals)")
    print("Speedup:     \(String(format: "%.1f", speedup))x")
    print("Target:      analytical < 2s")

    // Analytical must be at least 10x faster
    XCTAssertGreaterThan(speedup, 10.0, "Analytical should be ≥10x faster")
    // Analytical should be under 5s on simulator
    XCTAssertLessThan(agTime, 5.0, "Analytical should complete in <5s")
}
```

**Expected output**:
```
Finite-diff: ~47s (96659 evals)
Analytical:  ~1-2s (~200 evals)
Speedup:     ~25-50x
```

**12. Full pipeline end-to-end with pixel comparison**
```swift
func testFullPipelinePixelMatch() throws {
    // Run full DewarpPipeline.process() with analytical gradient
    // Compare output image against:
    //   (a) Finite-diff L-BFGS-B output — should be near-identical (same optimizer, same minimum)
    //   (b) Python reference output — should match as well as finite-diff does
    //
    // This catches any regression from switching to pure-Swift projection.
    // If the pure-Swift projectPoints has any discrepancy from OpenCV,
    // it would show up as different output pixels.

    let image = try GoldenFileLoader.loadImage("boston_cooking_a_input.jpg")
    let result = DewarpPipeline.process(image: image)
    guard case .success(let output) = result else {
        XCTFail("Pipeline failed"); return
    }

    // Output dimensions should be reasonable
    let w = Int(output.size.width * output.scale)
    let h = Int(output.size.height * output.scale)
    XCTAssertGreaterThan(w, 100)
    XCTAssertGreaterThan(h, 100)
    print("Output dimensions: \(w) x \(h)")

    // Attach for visual inspection
    let attachment = XCTAttachment(image: output)
    attachment.name = "analytical_gradient_output"
    attachment.lifetime = .keepAlways
    add(attachment)
}
```

**13. Comparison against Python reference on comparison/ input**
```swift
func testCompareWithPythonReferenceImage() throws {
    // Use comparison/input.jpg (has matching Python output)
    // Run pipeline with analytical gradient
    // Compare pixel-by-pixel with comparison/python.png
    // Log dimension match and pixel match percentages
    //
    // This validates the entire chain: pure-Swift projection + analytical gradient
    // + L-BFGS-B produces output that matches the Python reference.

    // Expected: dimension match ~95%+, pixel match at cropped region ~90%+
    // (same as or better than finite-diff, since we find the same minimum)
}
```

### Regression Tests

All existing tests must continue to pass:
- `PowellOptimizerTests` (Powell still used for page-dim)
- `LBFGSBOptimizerTests` (finite-diff overload unchanged)
- `PipelineIntegrationTests` (now uses analytical gradient)

## Performance Considerations

### Expected improvement

| Metric | Finite diff (current) | Analytical (projected) |
|--------|----------------------|----------------------|
| Evals per iteration | N+1 ≈ 593 | 1 (combined f+g) |
| Total evaluations | ~97K | ~200 |
| Bridge crossings | ~97K | 0 (pure Swift) |
| Time per image | 47s | <2s (projected) |

### Memory

The Jacobian computation adds per-point temporary storage:
- Per point: 6 doubles for each of dProj/dPoint, dProj/dRvec, dProj/dTvec = 18 doubles
- Total: ~600 points × 18 × 8 bytes ≈ 86KB — negligible

### Numerical stability

Rodrigues rotation is numerically stable except at θ ≈ 0 (identity rotation). Use the small-angle approximation `R ≈ I + [r]×` when `θ < 1e-10`, matching OpenCV's behavior. The initial parameters from `solvePnP` typically have `||rvec|| ≈ 0.07`, well away from zero.

## Security Considerations

None. Pure numerical computation.

## Documentation

- Update `docs/differences.md` section 3 to note Swift uses analytical gradients
- The remaining difference from Python becomes: same algorithm, same gradient approach (analytical), but hand-coded chain rule instead of JAX autodiff

## Implementation Phases

### Phase 1: Pure-Swift Projection
- Implement `rodrigues()` with full 9×3 Jacobian (section 3 formulas)
- Implement `projectAndDifferentiate()` with per-point Jacobians
- Unit tests 1–5 (Rodrigues accuracy, Jacobian vs finite diff, match OpenCV)

### Phase 2: Analytical Gradient
- Implement `objectiveAndGradient()` using accumulation patterns from section 7
- Unit tests 6–9 (cubic derivatives, clamp boundary, full gradient vs finite diff)

### Phase 3: Integration and Validation
- Add `lbfgsbMinimize(objectiveAndGradient:)` overload
- Switch `DewarpPipeline.swift` to use it
- Integration tests 10–13 (side-by-side comparison, timing, pixel match)
- Performance measurement and documentation update

## Open Questions

1. **Should we keep the finite-difference overload?** Yes — keep `lbfgsbMinimize(objective:)` for the page-dimension optimization (only 2 params, where analytical gradient is overkill) and as a debugging fallback.

2. **Rodrigues edge case at θ = 0**: When the rotation vector is zero (identity), the Jacobian has a removable singularity. OpenCV uses a Taylor expansion. We should match this. The initial parameters from `solvePnP` typically have `||rvec|| ≈ 0.07`, well away from zero, but the small-angle branch must be correct for robustness.

## References

- OpenCV `projectPoints`: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga1019495a2c8d1743ed5cc23fa0daff8c
- Rodrigues formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
- OpenCV Rodrigues implementation: `modules/calib3d/src/calibration.cpp` (function `cvRodrigues2`, lines ~550–650)
- Python reference: `src/page_dewarp/projection.py` and `src/page_dewarp/optimise/_jax.py`
- Current pure-Swift cubic: `Sources/Core/Projection.swift:17–64`
- Current objective: `Sources/Core/Objective.swift`
- Current keypoint indexing: `Sources/Core/Keypoints.swift:51–57`
- Camera matrix: `Sources/Core/CameraMatrix.swift` — `K = [[1.2, 0, 0], [0, 1.2, 0], [0, 0, 1]]`
