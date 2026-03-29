// PowellOptimizerTests.swift
// Tests for powellMinimize() and brentMinimize1D()

import XCTest
@testable import PageDewarp

final class PowellOptimizerTests: XCTestCase {

    // MARK: - Brent 1D minimization tests

    func testBrentMinimizesSimpleParabola() {
        // f(x) = (x - 3)^2, minimum at x=3
        let f: (Double) -> Double = { x in (x - 3) * (x - 3) }
        let (x, fx, _) = brentMinimize1D(f: f, ax: 0.0, bx: 2.0, cx: 6.0)
        XCTAssertEqual(x, 3.0, accuracy: 1e-6, "Brent should find minimum at x=3")
        XCTAssertEqual(fx, 0.0, accuracy: 1e-10, "f(3) should be 0")
    }

    func testBrentMinimizesCosine() {
        // f(x) = cos(x), minimum near x = pi in [2, 4]
        let f: (Double) -> Double = { x in cos(x) }
        let (x, fx, _) = brentMinimize1D(f: f, ax: 2.0, bx: 3.0, cx: 4.0)
        XCTAssertEqual(x, Double.pi, accuracy: 1e-6, "Brent should find minimum of cos at pi")
        XCTAssertEqual(fx, -1.0, accuracy: 1e-8, "cos(pi) = -1")
    }

    // MARK: - Powell 2D quadratic

    func testConvergesOnSimpleQuadratic2D() {
        // f(x, y) = (x - 3)^2 + (y - 4)^2, minimum at (3, 4)
        let objective: ([Double]) -> Double = { p in
            let dx = p[0] - 3.0
            let dy = p[1] - 4.0
            return dx * dx + dy * dy
        }
        let result = powellMinimize(objective: objective, x0: [0.0, 0.0])
        XCTAssertEqual(result.x[0], 3.0, accuracy: 1e-4, "x should converge to 3")
        XCTAssertEqual(result.x[1], 4.0, accuracy: 1e-4, "y should converge to 4")
        XCTAssertEqual(result.fun, 0.0, accuracy: 1e-6, "minimum value should be 0")
        XCTAssertTrue(result.converged, "should mark as converged")
    }

    // MARK: - Powell Rosenbrock

    func testConvergesOnRosenbrock() {
        // Rosenbrock: f(x,y) = (1-x)^2 + 100*(y-x^2)^2, minimum at (1,1)
        // Ported validation from task P2.4 spec
        let objective: ([Double]) -> Double = { p in
            let x = p[0], y = p[1]
            let a = 1.0 - x
            let b = y - x * x
            return a * a + 100.0 * b * b
        }
        let result = powellMinimize(objective: objective, x0: [0.0, 0.0])
        XCTAssertEqual(result.x[0], 1.0, accuracy: 1e-3, "Rosenbrock x should converge to 1")
        XCTAssertEqual(result.x[1], 1.0, accuracy: 1e-3, "Rosenbrock y should converge to 1")
        XCTAssertLessThan(result.fun, 1e-6, "Rosenbrock minimum value should be near 0")
    }

    // MARK: - Powell 10D quadratic

    func testConvergesOn10DQuadratic() {
        // f(x) = sum_i (x_i - target_i)^2, minimum at target
        let target = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        let objective: ([Double]) -> Double = { p in
            zip(p, target).reduce(0.0) { acc, pair in
                let diff = pair.0 - pair.1
                return acc + diff * diff
            }
        }
        let x0 = [Double](repeating: 0.0, count: 10)
        let result = powellMinimize(objective: objective, x0: x0)
        XCTAssertLessThan(result.fun, 1e-6, "10D quadratic minimum should be near 0")
        for i in 0..<10 {
            XCTAssertEqual(result.x[i], target[i], accuracy: 1e-4, "x[\(i)] should match target")
        }
        // Nfev should not hit maxIter for a simple quadratic
        XCTAssertTrue(result.converged, "10D quadratic should converge")
    }

    // MARK: - nfev sanity

    func testNfevIsPositive() {
        let objective: ([Double]) -> Double = { p in p[0] * p[0] }
        let result = powellMinimize(objective: objective, x0: [5.0])
        XCTAssertGreaterThan(result.nfev, 0, "nfev should be positive after optimization")
    }

    // MARK: - OptimizeResult fields

    func testOptimizeResultHasExpectedFields() {
        let objective: ([Double]) -> Double = { p in
            (p[0] - 2.0) * (p[0] - 2.0)
        }
        let result = powellMinimize(objective: objective, x0: [0.0])
        XCTAssertEqual(result.x.count, 1, "result.x should have same length as x0")
        XCTAssertFalse(result.fun.isNaN, "result.fun should not be NaN")
        XCTAssertFalse(result.fun.isInfinite, "result.fun should not be Inf")
        XCTAssertGreaterThanOrEqual(result.nfev, 1)
    }
}
