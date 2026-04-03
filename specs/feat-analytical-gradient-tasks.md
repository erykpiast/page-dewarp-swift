# Task Breakdown: Analytical Gradient for L-BFGS-B

Generated: 2026-04-03
Source: specs/feat-analytical-gradient.md

## Overview

Replace finite-difference numerical gradients with analytical gradients computed via chain rule. Requires: (1) pure-Swift Rodrigues + pinhole projection with Jacobians, (2) chain-rule gradient assembly, (3) new L-BFGS-B overload, (4) real-case validation against finite-diff and Python reference.

## Phase 1: Pure-Swift Projection

### Task 1.1: Implement `PureProjection.swift` — Rodrigues + pinhole with Jacobians
**Description**: Pure-Swift reimplementation of `cv::projectPoints` with analytical Jacobians for the optimization loop.
**Size**: Large
**Priority**: High
**Dependencies**: None
**Can run parallel with**: Nothing (foundation)

**Implementation**: Create `Sources/Core/PureProjection.swift` with two functions:

1. `rodrigues(_ rvec: [Double]) -> (R: [Double], dR_dr: [Double])`

Rodrigues rotation formula:
```
θ = ||r||
k = r / θ
K = [0, -k₃, k₂; k₃, 0, -k₁; -k₂, k₁, 0]
R = I + sin(θ)·K + (1 - cos(θ))·K²
```

Small-angle case (θ < 1e-10): `R ≈ I + [r]×`, with Jacobian:
```
∂R/∂r₁ = [0,0,0; 0,0,-1; 0,1,0]
∂R/∂r₂ = [0,0,1; 0,0,0; -1,0,0]
∂R/∂r₃ = [0,-1,0; 1,0,0; 0,0,0]
```

General-case Jacobian (27 entries, 9×3 matrix stored flat):
```
∂R_ij/∂r_m = -sin(θ)·k_m·δ_ij
            + sin(θ)·k_m·(k_i·k_j)
            + (1-cos(θ))·(∂k_i/∂r_m·k_j + k_i·∂k_j/∂r_m)
            + cos(θ)·k_m·[k]×_ij
            + sin(θ)·∂[k]×_ij/∂r_m

where ∂k_i/∂r_m = (δ_im - k_i·k_m) / θ
```

Reference: OpenCV `cvRodrigues2` in `modules/calib3d/src/calibration.cpp` lines ~550–650.

2. `projectAndDifferentiate(points3D:rvec:tvec:focalLength:)`

For each 3D point P = (X, Y, Z):
```swift
// Camera space
let cx = R[0]*X + R[1]*Y + R[2]*Z + tx
let cy = R[3]*X + R[4]*Y + R[5]*Z + ty
let cz = R[6]*X + R[7]*Y + R[8]*Z + tz

// Perspective projection
let iz = 1.0 / cz
let u = focalLength * cx * iz
let v = focalLength * cy * iz

// Perspective Jacobian J_persp (2×3):
// J00 = f/cz,  J01 = 0,      J02 = -f*cx/cz²
// J10 = 0,     J11 = f/cz,   J12 = -f*cy/cz²

// dProj/dPoint (2×3): J_persp · R (rows of R for each camera axis)
// dProj/dRvec (2×3): J_persp · [dR/dr_m · P] for m=0,1,2
// dProj/dTvec (2×3): J_persp · I = J_persp itself
```

All Jacobians stored as flat `[Double]` arrays (6 elements per point for each 2×3 matrix).

No distortion (distCoeffs = 0), no principal point (cx = cy = 0). Camera matrix K = [[1.2, 0, 0], [0, 1.2, 0], [0, 0, 1]].

**Acceptance Criteria**:
- [ ] `rodrigues([0.1, 0.2, 0.3])` matches `OpenCVWrapper.rodriguesFromVector` to 1e-10
- [ ] `rodrigues([1e-12, 0, 0])` doesn't produce NaN (small-angle branch)
- [ ] Rodrigues Jacobian matches finite-difference Jacobian of `rodrigues()` to 1e-6
- [ ] `projectAndDifferentiate` output matches `OpenCVWrapper.projectPointsWith3DPoints` to 1e-8 on 100 random points
- [ ] Projection Jacobians match finite-difference Jacobians to 1e-5

---

### Task 1.2: Unit tests for PureProjection
**Description**: Write `Tests/PureProjectionTests.swift` with 5 tests validating Rodrigues and projection against OpenCV and finite differences.
**Size**: Medium
**Priority**: High
**Dependencies**: Task 1.1
**Can run parallel with**: Nothing

**Tests**:

```swift
import XCTest
@testable import PageDewarp

final class PureProjectionTests: XCTestCase {

    // Test 1: Rodrigues rotation matches OpenCV
    func testRodriguesMatchesOpenCV() {
        let rvec = [0.1, 0.2, 0.3]
        let (R, _) = rodrigues(rvec)
        let opencvR = OpenCVWrapper.rodriguesFromVector(rvec.map { NSNumber(value: $0) })
            .map { $0.doubleValue }
        for i in 0..<9 {
            XCTAssertEqual(R[i], opencvR[i], accuracy: 1e-10,
                "R[\(i)] mismatch: swift=\(R[i]) opencv=\(opencvR[i])")
        }
    }

    // Test 2: Rodrigues small-angle doesn't NaN
    func testRodriguesSmallAngle() {
        let (R, dR) = rodrigues([1e-12, 0, 0])
        XCTAssertFalse(R.contains(where: { $0.isNaN }))
        XCTAssertFalse(dR.contains(where: { $0.isNaN }))
        // Should be approximately identity
        XCTAssertEqual(R[0], 1.0, accuracy: 1e-8) // R[0,0]
        XCTAssertEqual(R[4], 1.0, accuracy: 1e-8) // R[1,1]
        XCTAssertEqual(R[8], 1.0, accuracy: 1e-8) // R[2,2]
    }

    // Test 3: Rodrigues Jacobian vs finite differences
    func testRodriguesJacobian() {
        let rvec = [0.1, 0.2, 0.3]
        let (_, dR_analytical) = rodrigues(rvec)
        let h = 1e-7
        for m in 0..<3 {
            var rp = rvec
            rp[m] += h
            let (Rp, _) = rodrigues(rp)
            let (R0, _) = rodrigues(rvec)
            for i in 0..<9 {
                let fd = (Rp[i] - R0[i]) / h
                let an = dR_analytical[i * 3 + m]  // 9×3 row-major
                XCTAssertEqual(an, fd, accuracy: 1e-5,
                    "dR[\(i)]/dr[\(m)] mismatch: analytical=\(an) fd=\(fd)")
            }
        }
    }

    // Test 4: Pure projection matches OpenCV
    func testProjectionMatchesOpenCV() {
        let rvec = [-0.057, 0.071, 0.011]  // from golden file
        let tvec = [-0.605, -0.958, 1.218]
        let f = 1.2
        // Generate 20 test points
        var points3D: [Double] = []
        for i in 0..<20 {
            let x = Double(i) * 0.05
            let y = Double(i % 5) * 0.1
            let z = 0.01 * x * x
            points3D.append(contentsOf: [x, y, z])
        }
        let (projected, _, _, _) = projectAndDifferentiate(
            points3D: points3D, rvec: rvec, tvec: tvec, focalLength: f)
        // Compare with OpenCV
        let pts3D = points3D.enumerated().map { NSNumber(value: $0.element) }
        let opencvPts = OpenCVWrapper.projectPointsWith3DPoints(
            pts3D,
            rvec: rvec.map { NSNumber(value: $0) },
            tvec: tvec.map { NSNumber(value: $0) },
            cameraMatrix: [f,0,0, 0,f,0, 0,0,1].map { NSNumber(value: $0) },
            distCoeffs: [0,0,0,0,0].map { NSNumber(value: $0) })
        for i in 0..<20 {
            let pt = opencvPts[i].cgPointValue
            XCTAssertEqual(projected[i*2], Double(pt.x), accuracy: 1e-8)
            XCTAssertEqual(projected[i*2+1], Double(pt.y), accuracy: 1e-8)
        }
    }

    // Test 5: Projection Jacobians vs finite differences
    func testProjectionJacobians() {
        let rvec = [-0.057, 0.071, 0.011]
        let tvec = [-0.605, -0.958, 1.218]
        let f = 1.2
        let points3D = [0.3, 0.2, 0.01]  // single point
        let (_, dP, dR, dT) = projectAndDifferentiate(
            points3D: points3D, rvec: rvec, tvec: tvec, focalLength: f)
        let h = 1e-7
        // Check dProj/dRvec via finite diff
        for m in 0..<3 {
            var rp = rvec; rp[m] += h
            let (pp, _, _, _) = projectAndDifferentiate(
                points3D: points3D, rvec: rp, tvec: tvec, focalLength: f)
            let (p0, _, _, _) = projectAndDifferentiate(
                points3D: points3D, rvec: rvec, tvec: tvec, focalLength: f)
            let du_fd = (pp[0] - p0[0]) / h
            let dv_fd = (pp[1] - p0[1]) / h
            XCTAssertEqual(dR[m], du_fd, accuracy: 1e-5)       // du/dr_m
            XCTAssertEqual(dR[3+m], dv_fd, accuracy: 1e-5)     // dv/dr_m
        }
        // Similarly check dProj/dTvec and dProj/dPoint
    }
}
```

**Acceptance Criteria**:
- [ ] All 5 tests pass
- [ ] Rodrigues matches OpenCV to 1e-10
- [ ] Jacobians match finite differences to 1e-5

---

## Phase 2: Analytical Gradient

### Task 2.1: Implement `AnalyticalGradient.swift`
**Description**: Chain-rule gradient assembly using projection Jacobians from PureProjection.
**Size**: Large
**Priority**: High
**Dependencies**: Task 1.1
**Can run parallel with**: Task 1.2 (tests)

**Implementation**: Create `Sources/Core/AnalyticalGradient.swift` with `objectiveAndGradient()`.

The gradient accumulation for each keypoint k, given error gradient `eu = 2*(u-dst_u)`, `ev = 2*(v-dst_v)` and perspective Jacobian values `J00 = f/cz`, `J02 = -f*cx/cz²`, `J11 = f/cz`, `J12 = -f*cy/cz²`:

**rvec** (indices 0–2):
```swift
for m in 0..<3 {
    let dcx = dR_dr[0*3+m]*X + dR_dr[1*3+m]*Y + dR_dr[2*3+m]*Z
    let dcy = dR_dr[3*3+m]*X + dR_dr[4*3+m]*Y + dR_dr[5*3+m]*Z
    let dcz = dR_dr[6*3+m]*X + dR_dr[7*3+m]*Y + dR_dr[8*3+m]*Z
    let du_dr = J00*dcx + J02*dcz
    let dv_dr = J11*dcy + J12*dcz
    grad[m] += eu*du_dr + ev*dv_dr
}
```

**tvec** (indices 3–5):
```swift
grad[3] += eu * J00                  // du/dtx = f/cz
grad[4] += ev * J11                  // dv/dty = f/cz
grad[5] += eu * J02 + ev * J12      // d(u,v)/dtz
```

**cubic α, β** (indices 6–7):
```swift
let dz_dalpha = (abs(pvec[6]) >= 0.5) ? 0.0 : (x*x*x - 2*x*x + x)
let dz_dbeta  = (abs(pvec[7]) >= 0.5) ? 0.0 : (x*x*x - x*x)
let du_dz = J00*R[2] + J02*R[8]     // R column 2: (R02, R12, R22)
let dv_dz = J11*R[5] + J12*R[8]
grad[6] += (eu*du_dz + ev*dv_dz) * dz_dalpha
grad[7] += (eu*du_dz + ev*dv_dz) * dz_dbeta
```

**ycoords** (indices 8..8+nspans):
```swift
let du_dy = J00*R[1] + J02*R[7]     // R column 1
let dv_dy = J11*R[4] + J12*R[7]
grad[yIdx] += eu*du_dy + ev*dv_dy
```

**xcoords** (indices 8+nspans..end):
```swift
let dz_dx = 3*(alpha+beta)*x*x - 2*(2*alpha+beta)*x + alpha
let du_dx = J00*(R[0] + R[2]*dz_dx) + J02*(R[6] + R[8]*dz_dx)
let dv_dx = J11*(R[3] + R[5]*dz_dx) + J12*(R[6] + R[8]*dz_dx)
grad[xIdx] += eu*du_dx + ev*dv_dx
```

**Shear penalty**: `grad[0] += 2 * shearCost * pvec[0]`

The function computes both `f` and `grad` in a single pass over all keypoints.

**Acceptance Criteria**:
- [ ] `objectiveAndGradient` compiles and returns (f, grad) tuple
- [ ] Gradient matches `finiteDifferenceGradient` on the real dewarp objective to 1e-4 (test 8)
- [ ] Clamp boundary: grad[6]=0 when |pvec[6]|>=0.5

---

### Task 2.2: Unit tests for AnalyticalGradient
**Description**: Write `Tests/AnalyticalGradientTests.swift` — the critical gradient-vs-finite-diff test plus cubic and smoke tests.
**Size**: Medium
**Priority**: High
**Dependencies**: Task 2.1
**Can run parallel with**: Nothing

**Tests**:

```swift
final class AnalyticalGradientTests: XCTestCase {

    // Test 6: Cubic polynomial derivatives
    func testCubicDerivatives() {
        let x = 0.3, alpha = 0.15, beta = 0.1
        let h = 1e-7
        let a = alpha + beta, b = -2*alpha - beta, c = alpha

        // dz/dx
        let z0 = ((a*x + b)*x + c)*x
        let z1 = ((a*(x+h) + b)*(x+h) + c)*(x+h)
        let dz_dx_fd = (z1 - z0) / h
        let dz_dx_an = 3*(alpha+beta)*x*x - 2*(2*alpha+beta)*x + alpha
        XCTAssertEqual(dz_dx_an, dz_dx_fd, accuracy: 1e-5)

        // dz/dalpha
        let a1 = (alpha+h) + beta, b1 = -2*(alpha+h) - beta, c1 = (alpha+h)
        let z_a1 = ((a1*x + b1)*x + c1)*x
        let dz_da_fd = (z_a1 - z0) / h
        let dz_da_an = x*x*x - 2*x*x + x
        XCTAssertEqual(dz_da_an, dz_da_fd, accuracy: 1e-5)

        // dz/dbeta
        let a2 = alpha + (beta+h), b2 = -2*alpha - (beta+h)
        let z_b1 = ((a2*x + b2)*x + c)*x
        let dz_db_fd = (z_b1 - z0) / h
        let dz_db_an = x*x*x - x*x
        XCTAssertEqual(dz_db_an, dz_db_fd, accuracy: 1e-5)
    }

    // Test 7: Clamp boundary — gradient is zero
    func testClampBoundary() {
        // pvec with alpha at +0.5 (clamped)
        var pvec = [Double](repeating: 0.0, count: 20)
        pvec[6] = 0.5  // alpha at boundary
        pvec[7] = 0.1  // beta not clamped
        // Minimal setup with 1 keypoint
        let dstpoints = [[0.0, 0.0], [0.1, 0.1]]
        let keypointIndex = [[0, 0], [9, 8]]
        pvec[8] = 0.1  // y
        pvec[9] = 0.2  // x
        let (_, grad) = objectiveAndGradient(
            pvec: pvec, dstpoints: dstpoints, keypointIndex: keypointIndex,
            shearCost: 0.0, focalLength: 1.2)
        XCTAssertEqual(grad[6], 0.0, accuracy: 1e-15, "Clamped alpha gradient should be zero")
    }

    // Test 8: CRITICAL — Full gradient vs finite differences on golden file data
    func testGradientMatchesFiniteDifferences() throws {
        let image = try GoldenFileLoader.loadImage("boston_cooking_a_input.jpg")
        // Reconstruct pipeline intermediates to get dstpoints, keypointIndex, initialParams
        // (same setup as EvalComparisonTests.testOptimizerDiagnostics)
        // ...

        let objective = makeObjective(
            dstpoints: dstpoints, keypointIndex: keypointIndex,
            shearCost: DewarpConfig.shearCost, rvecRange: DewarpConfig.rvecIdx)

        let (f_an, grad_an) = objectiveAndGradient(
            pvec: initialParams, dstpoints: dstpoints, keypointIndex: keypointIndex,
            shearCost: DewarpConfig.shearCost, focalLength: DewarpConfig.focalLength)

        let f_fd = objective(initialParams)
        let grad_fd = finiteDifferenceGradient(
            objective: objective, x: initialParams, f0: f_fd)

        // Objective values must match (both use same math, different gradient)
        XCTAssertEqual(f_an, f_fd, accuracy: 1e-10, "Objective values must match exactly")

        // Gradient comparison — element by element
        XCTAssertEqual(grad_an.count, grad_fd.count)
        var maxRelErr = 0.0
        for i in 0..<grad_an.count {
            let denom = max(abs(grad_fd[i]), 1e-8)
            let relErr = abs(grad_an[i] - grad_fd[i]) / denom
            maxRelErr = max(maxRelErr, relErr)
            XCTAssertEqual(grad_an[i], grad_fd[i], accuracy: max(1e-4, abs(grad_fd[i]) * 0.01),
                "grad[\(i)] mismatch: analytical=\(grad_an[i]) fd=\(grad_fd[i])")
        }
        print("Max relative gradient error: \(maxRelErr)")
    }

    // Test 9: Gradient is nonzero at initial params
    func testGradientNonzeroAtInitialParams() throws {
        // Same setup as test 8, but just check gradient magnitude
        let maxGrad = grad_an.map { abs($0) }.max() ?? 0
        XCTAssertGreaterThan(maxGrad, 0.01, "Gradient should be meaningful at initial params")
    }
}
```

**Acceptance Criteria**:
- [ ] Test 8 passes (gradient matches finite diff to 1e-4)
- [ ] Cubic derivatives match to 1e-5
- [ ] Clamp boundary gradient is zero

---

## Phase 3: Integration and Validation

### Task 3.1: Add L-BFGS-B `objectiveAndGradient` overload
**Description**: Add a second `lbfgsbMinimize` overload that accepts a combined objective+gradient function, eliminating finite differences.
**Size**: Small
**Priority**: High
**Dependencies**: Task 2.1
**Can run parallel with**: Task 2.2

**Implementation**: Add to `Sources/Core/LBFGSBOptimizer.swift`:

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
    // Identical to existing lbfgsbMinimize, except the "FG" branch:
    //   let (fVal, gVal) = objectiveAndGradient(Array(x))
    //   f = fVal
    //   for i in 0..<n { g[i] = gVal[i] }
    //   nfev += 1  // (not N+1)
}
```

The existing `lbfgsbMinimize(objective:)` overload is kept unchanged.

**Acceptance Criteria**:
- [ ] New overload compiles
- [ ] Existing `LBFGSBOptimizerTests` still pass
- [ ] New overload minimizes Rosenbrock with a hand-coded gradient

---

### Task 3.2: Switch DewarpPipeline and run real-case validation
**Description**: Wire analytical gradient into the pipeline and run side-by-side comparison against finite-diff and Python reference.
**Size**: Medium
**Priority**: High
**Dependencies**: Tasks 2.2, 3.1

**Implementation**: Change `DewarpPipeline.swift` line 109:

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

**Validation tests** (in `Tests/AnalyticalGradientIntegrationTests.swift`):

Test 10 — Side-by-side optimizer comparison:
- Run both finite-diff and analytical on golden file
- Assert both converge
- Assert analytical nfev < 1000 (vs ~97K for finite-diff)
- Assert final loss within 5% relative
- Assert rvec/tvec/cubic match within 0.05
- Print comparison table including Python reference values

Test 11 — Wall-clock timing:
- Time both optimizers on same input
- Assert analytical ≥10x faster
- Assert analytical < 5s on simulator
- Print: FD time, analytical time, speedup factor

Test 12 — Full pipeline end-to-end:
- Run `DewarpPipeline.process()` on golden file
- Assert success, reasonable output dimensions
- Attach output image for visual inspection

Test 13 — Python reference pixel comparison:
- Run pipeline on `comparison/input.jpg`
- Compare dimensions and pixels against `comparison/python.png`
- Log dimension match % and pixel match %

**Acceptance Criteria**:
- [ ] Pipeline produces valid output
- [ ] Analytical optimizer uses <1000 function evaluations
- [ ] Speedup ≥10x over finite-diff
- [ ] Final loss within 5% of finite-diff result
- [ ] All existing tests still pass

---

### Task 3.3: Update documentation
**Description**: Update `docs/differences.md` section 3 to reflect analytical gradients.
**Size**: Small
**Priority**: Medium
**Dependencies**: Task 3.2

Update section 3 to note:
- Swift now uses analytical (hand-coded chain rule) gradients
- Same L-BFGS-B algorithm as Python
- Remaining difference: hand-coded derivatives vs JAX autodiff (functionally equivalent)
- Performance now comparable to Python

**Acceptance Criteria**:
- [ ] `docs/differences.md` accurately describes the analytical gradient approach
