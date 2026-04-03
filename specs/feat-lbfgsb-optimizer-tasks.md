# Task Breakdown: Replace Powell with L-BFGS-B Optimizer

Generated: 2026-04-03
Source: specs/feat-lbfgsb-optimizer.md

## Overview

Replace the Powell optimizer with L-BFGS-B by vendoring the `stephenbeckr/L-BFGS-B-C` library (same C code SciPy uses), writing a thin Swift wrapper with finite-difference gradient computation, and swapping the single optimizer call in `DewarpPipeline.swift`.

## Phase 1: Vendor and Integrate L-BFGS-B-C

### Task 1.1: Vendor L-BFGS-B-C source files
**Description**: Download and vendor `lbfgsb.c`, `miniCBLAS.c`, and their headers from `stephenbeckr/L-BFGS-B-C` into the project.
**Size**: Small
**Priority**: High
**Dependencies**: None
**Can run parallel with**: Nothing (foundation task)

**Implementation Steps**:
1. Clone or download `stephenbeckr/L-BFGS-B-C` from https://github.com/stephenbeckr/L-BFGS-B-C
2. Copy these files into `Sources/LBFGSB/`:
   - `src/lbfgsb.c`
   - `src/lbfgsb.h`
   - `src/miniCBLAS.c`
   - `src/miniCBLAS.h`
3. Copy the BSD-3 LICENSE file into `Sources/LBFGSB/LICENSE`
4. Create `Sources/LBFGSB/module.modulemap`:
   ```
   module LBFGSB {
       header "lbfgsb.h"
       export *
   }
   ```
5. Verify the project builds with `xcodegen generate && xcodebuild -scheme PageDewarp -sdk iphonesimulator build`

**Acceptance Criteria**:
- [ ] `Sources/LBFGSB/` contains `lbfgsb.c`, `lbfgsb.h`, `miniCBLAS.c`, `miniCBLAS.h`, `LICENSE`, `module.modulemap`
- [ ] BSD-3 LICENSE file is present
- [ ] Project compiles with the new C files (no build errors)
- [ ] `import LBFGSB` works from a Swift file in the `PageDewarp` target

---

### Task 1.2: Implement Swift wrapper for `setulb()`
**Description**: Create `LBFGSBOptimizer.swift` with a clean Swift API that drives the C `setulb()` reverse-communication loop and computes finite-difference gradients.
**Size**: Medium
**Priority**: High
**Dependencies**: Task 1.1
**Can run parallel with**: Nothing

**Technical Requirements**:

The C `setulb()` function signature:
```c
void setulb_(int *n, int *m, double *x, double *l, double *u, int *nbd,
             double *f, double *g, double *factr, double *pgtol,
             double *wa, int *iwa, char *task, int *iprint,
             char *csave, int *lsave, int *isave, double *dsave);
```

Working array sizes:
- `wa[(2*m + 5)*n + 12*m*(m+1)]` — for m=100, n=400: ~1.3MB
- `iwa[3*n]` — for n=400: ~5KB
- `csave[60]`, `lsave[4]`, `isave[44]`, `dsave[29]`

All parameters unconstrained: `nbd[i] = 0` for all i.

**Implementation Steps**:

1. Create `Sources/Core/LBFGSBOptimizer.swift` with:

```swift
import Foundation
import LBFGSB

/// Compute gradient via forward finite differences.
/// Cost: N function evaluations (reuses f(x) passed in as f0).
func finiteDifferenceGradient(
    objective: ([Double]) -> Double,
    x: [Double],
    f0: Double,
    epsilon: Double = 1e-8
) -> [Double] {
    let n = x.count
    var grad = [Double](repeating: 0.0, count: n)
    var xp = x
    for i in 0..<n {
        let xi = x[i]
        let h = max(epsilon, epsilon * abs(xi))
        xp[i] = xi + h
        let fi = objective(xp)
        grad[i] = (fi - f0) / h
        xp[i] = xi
    }
    return grad
}

/// Minimizes a scalar function using L-BFGS-B via the vendored C library.
///
/// Uses finite-difference numerical gradients. Parameters are unconstrained.
/// Matches SciPy's `minimize(method='L-BFGS-B')` configuration.
///
/// - Parameters:
///   - objective: Scalar objective function `([Double]) -> Double`.
///   - x0: Initial parameter vector.
///   - maxIter: Maximum number of iterations (default 600,000).
///   - maxCor: Number of L-BFGS correction pairs (default 100).
///   - ftol: Function value tolerance factor (default: machine epsilon).
///   - gtol: Projected gradient tolerance (default 1e-5).
///   - epsilon: Finite difference step size (default 1e-8).
/// - Returns: Optimization result.
func lbfgsbMinimize(
    objective: ([Double]) -> Double,
    x0: [Double],
    maxIter: Int = 600_000,
    maxCor: Int = 100,
    ftol: Double = 2.220446049250313e-16,
    gtol: Double = 1e-5,
    epsilon: Double = 1e-8
) -> OptimizeResult {
    var n = Int32(x0.count)
    var m = Int32(maxCor)

    // Parameter vector (in/out)
    var x = x0

    // Bounds — all unconstrained
    var l = [Double](repeating: 0.0, count: x0.count)
    var u = [Double](repeating: 0.0, count: x0.count)
    var nbd = [Int32](repeating: 0, count: x0.count)

    // Function value and gradient (in/out)
    var f: Double = 0.0
    var g = [Double](repeating: 0.0, count: x0.count)

    // Tolerances
    var factr = ftol / 2.220446049250313e-16  // factr * macheps = ftol
    var pgtol = gtol

    // Working arrays
    let waSize = (2 * Int(m) + 5) * Int(n) + 12 * Int(m) * (Int(m) + 1)
    var wa = [Double](repeating: 0.0, count: waSize)
    var iwa = [Int32](repeating: 0, count: 3 * Int(n))

    // Task string (60 chars, Fortran-style blank-padded)
    var task = [CChar](repeating: 0x20, count: 60)  // spaces
    // Write "START" into task
    let startBytes: [CChar] = [0x53, 0x54, 0x41, 0x52, 0x54]  // "START"
    for i in 0..<startBytes.count { task[i] = startBytes[i] }

    // Internal state arrays
    var csave = [CChar](repeating: 0x20, count: 60)
    var lsave = [Int32](repeating: 0, count: 4)
    var isave = [Int32](repeating: 0, count: 44)
    var dsave = [Double](repeating: 0.0, count: 29)

    var iprint: Int32 = -1  // silent
    var nfev = 0
    var nit = 0
    var converged = false

    for _ in 0..<(maxIter + 1) {
        setulb_(&n, &m, &x, &l, &u, &nbd,
                &f, &g, &factr, &pgtol,
                &wa, &iwa, &task, &iprint,
                &csave, &lsave, &isave, &dsave)

        let taskStr = String(cString: task.map { $0 == 0x20 ? 0 : $0 } + [0])
            .trimmingCharacters(in: .whitespaces)

        if taskStr.hasPrefix("FG") {
            // Evaluate function and gradient
            f = objective(Array(x))
            nfev += 1
            g = finiteDifferenceGradient(
                objective: objective, x: Array(x), f0: f, epsilon: epsilon
            )
            nfev += x0.count  // N evaluations for gradient
        } else if taskStr.hasPrefix("NEW_X") {
            nit += 1
            continue
        } else if taskStr.hasPrefix("CONV") {
            converged = true
            break
        } else if taskStr.hasPrefix("ABNO") || taskStr.hasPrefix("ERROR") {
            break
        } else {
            break
        }
    }

    return OptimizeResult(x: Array(x), fun: f, nfev: nfev, converged: converged)
}
```

2. Note: The exact `setulb_` function name may need adjustment — f2c-generated code typically appends an underscore. Check `lbfgsb.h` for the actual exported symbol name.

3. The task string parsing may need refinement — f2c Fortran strings are blank-padded, not null-terminated. The implementation above handles this but may need adjustment based on actual behavior.

**Acceptance Criteria**:
- [ ] `LBFGSBOptimizer.swift` compiles and links against the vendored C code
- [ ] `lbfgsbMinimize` successfully minimizes `f(x) = x[0]^2 + x[1]^2` from `(5, 3)` to near `(0, 0)`
- [ ] `finiteDifferenceGradient` returns correct gradient for `f(x) = x[0]^2 + x[1]^2` at `(3, 4)` → `[6, 8]` within tolerance 1e-4
- [ ] Returns `OptimizeResult` matching the existing struct from `PowellOptimizer.swift:11–16`
- [ ] No memory leaks (all working arrays are stack/value-type allocated)

---

### Task 1.3: Unit tests for L-BFGS-B wrapper
**Description**: Write unit tests validating the optimizer on standard benchmarks and edge cases.
**Size**: Medium
**Priority**: High
**Dependencies**: Task 1.2
**Can run parallel with**: Nothing

**Tests to implement in `Tests/LBFGSBOptimizerTests.swift`**:

```swift
import XCTest
@testable import PageDewarp

final class LBFGSBOptimizerTests: XCTestCase {

    // Test 1: Gradient accuracy
    // Purpose: catch off-by-one or step-size bugs in gradient computation
    func testFiniteDifferenceGradient() {
        let objective: ([Double]) -> Double = { x in
            x[0] * x[0] + x[1] * x[1]
        }
        let grad = finiteDifferenceGradient(
            objective: objective, x: [3.0, 4.0], f0: 25.0
        )
        XCTAssertEqual(grad[0], 6.0, accuracy: 1e-4)
        XCTAssertEqual(grad[1], 8.0, accuracy: 1e-4)
    }

    // Test 2: setulb() wrapper round-trip
    // Purpose: catch bridging bugs (array layout, task-string parsing, working array sizing)
    func testSimpleQuadratic() {
        let objective: ([Double]) -> Double = { x in
            x[0] * x[0] + x[1] * x[1]
        }
        let result = lbfgsbMinimize(objective: objective, x0: [5.0, 3.0])
        XCTAssertTrue(result.converged)
        XCTAssertEqual(result.x[0], 0.0, accuracy: 1e-4)
        XCTAssertEqual(result.x[1], 0.0, accuracy: 1e-4)
        XCTAssertGreaterThan(result.nfev, 0)
    }

    // Test 3: Rosenbrock function — standard nonlinear optimization benchmark
    // Purpose: validates the full optimizer on a non-trivial landscape
    // This test CAN fail if bridging or gradient has bugs
    func testRosenbrock() {
        let objective: ([Double]) -> Double = { x in
            let a = 1.0 - x[0]
            let b = x[1] - x[0] * x[0]
            return a * a + 100.0 * b * b
        }
        let result = lbfgsbMinimize(objective: objective, x0: [-1.0, 1.0])
        XCTAssertTrue(result.converged)
        XCTAssertEqual(result.x[0], 1.0, accuracy: 1e-3)
        XCTAssertEqual(result.x[1], 1.0, accuracy: 1e-3)
    }

    // Test 4: High-dimensional quadratic — mimics real parameter count
    // Purpose: validates scaling behavior with realistic N
    func testHighDimensionalQuadratic() {
        let n = 200
        let objective: ([Double]) -> Double = { x in
            var sum = 0.0
            for i in 0..<n {
                let d = x[i] - Double(i)
                sum += d * d
            }
            return sum
        }
        let x0 = [Double](repeating: 0.0, count: n)
        let result = lbfgsbMinimize(objective: objective, x0: x0)
        XCTAssertTrue(result.converged)
        for i in 0..<n {
            XCTAssertEqual(result.x[i], Double(i), accuracy: 1e-3)
        }
    }

    // Test 5: Convergence criteria — already-optimal starting point
    // Purpose: catch bugs where optimizer doesn't stop when it should
    func testAlreadyOptimal() {
        let objective: ([Double]) -> Double = { x in
            x[0] * x[0] + x[1] * x[1]
        }
        let result = lbfgsbMinimize(objective: objective, x0: [0.0, 0.0])
        XCTAssertTrue(result.converged)
        XCTAssertEqual(result.fun, 0.0, accuracy: 1e-10)
    }
}
```

**Acceptance Criteria**:
- [ ] All 5 tests pass
- [ ] Rosenbrock converges within 1e-3 tolerance
- [ ] High-dimensional (n=200) test converges
- [ ] Already-optimal test returns immediately with fun ≈ 0
- [ ] All existing `PowellOptimizerTests` still pass (regression check)

---

## Phase 2: Integration and Validation

### Task 2.1: Switch DewarpPipeline to L-BFGS-B
**Description**: Replace the Powell optimizer call with L-BFGS-B for the main parameter optimization in `DewarpPipeline.swift`.
**Size**: Small
**Priority**: High
**Dependencies**: Task 1.3
**Can run parallel with**: Nothing

**Implementation Steps**:

Single-line change in `DewarpPipeline.swift` at line 109:

```swift
// Before (line 109):
let optResult = powellMinimize(objective: objective, x0: initialParams)

// After:
let optResult = lbfgsbMinimize(objective: objective, x0: initialParams)
```

Powell remains unchanged for the 2D page-dimension optimization at line 217:
```swift
// UNCHANGED — only 2 parameters, Powell is fine
let result = powellMinimize(objective: dimObjective, x0: dims)
```

**Acceptance Criteria**:
- [ ] `DewarpPipeline.swift:109` calls `lbfgsbMinimize` instead of `powellMinimize`
- [ ] `DewarpPipeline.swift:217` still calls `powellMinimize` for page dimensions
- [ ] Project compiles without errors
- [ ] All existing `PowellOptimizerTests` still pass

---

### Task 2.2: Pipeline integration tests
**Description**: Run the full pipeline with L-BFGS-B and validate output against golden files and Python reference.
**Size**: Medium
**Priority**: High
**Dependencies**: Task 2.1
**Can run parallel with**: Nothing

**Tests to implement/update in `Tests/PipelineIntegrationTests.swift`**:

```swift
// Test 6: Pipeline golden file comparison
// Purpose: end-to-end validation that the optimizer swap doesn't break the pipeline
func testPipelineWithLBFGSB() {
    // Run DewarpPipeline.process on a test image
    // Verify output is non-nil
    // Verify output dimensions are reasonable (within 10% of expected)
    // Compare against golden output pixel hash
}

// Test 7: Python reference comparison
// Purpose: validates that L-BFGS-B actually closes the accuracy gap
// This is the KEY acceptance test for this feature
// Target: average pixel match > 93% (up from ~89% with Powell)
func testPixelMatchWithPythonReference() {
    // Run on 5+ test images
    // Compare pixel match with Python reference output
    // Assert average > 93%
}
```

**Acceptance Criteria**:
- [ ] Pipeline produces valid output on all test images (non-nil, reasonable dimensions)
- [ ] Golden file comparison shows output has changed (different from Powell output)
- [ ] Pixel match with Python reference improves over Powell baseline (~89%)
- [ ] Target: average pixel match > 93%
- [ ] Processing time is measurably faster than Powell (log times for comparison)

---

## Phase 3: Documentation

### Task 3.1: Update documentation
**Description**: Update `docs/differences.md` section 3 to reflect the optimizer change.
**Size**: Small
**Priority**: Medium
**Dependencies**: Task 2.2
**Can run parallel with**: Nothing

**Implementation Steps**:

Update `docs/differences.md` section "### 3. Optimizer: Powell vs L-BFGS-B" to reflect:
- Swift now uses L-BFGS-B via vendored `L-BFGS-B-C` (same C code as SciPy)
- Gradients computed via forward finite differences (not JAX autodiff)
- Expected performance: comparable to Python (within 2-5x, depending on parameter count)
- Expected pixel match: >93% (up from ~89% with Powell)
- Powell retained for 2D page-dimension optimization only

**Acceptance Criteria**:
- [ ] `docs/differences.md` section 3 accurately describes the new optimizer
- [ ] No stale references to "Powell for main optimization"
