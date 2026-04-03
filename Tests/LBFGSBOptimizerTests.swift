// LBFGSBOptimizerTests.swift

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

    // Test 2: setulb() wrapper round-trip on simple quadratic
    // Purpose: catch bridging bugs (array layout, task code parsing, working array sizing)
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
