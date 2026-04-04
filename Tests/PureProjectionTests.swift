// PureProjectionTests.swift

import XCTest
@testable import PageDewarp
#if SWIFT_PACKAGE
import OpenCVBridge
#endif

final class PureProjectionTests: XCTestCase {

    // Test 1: Rodrigues rotation matches OpenCV
    func testRodriguesMatchesOpenCV() {
        let rvec = [0.1, 0.2, 0.3]
        let (R, _) = rodrigues(rvec)
        let opencvR = OpenCVWrapper.rodrigues(fromVector:rvec.map { NSNumber(value: $0) })
            .map { $0.doubleValue }
        for i in 0..<9 {
            XCTAssertEqual(R[i], opencvR[i], accuracy: 1e-10,
                "R[\(i)] mismatch: swift=\(R[i]) opencv=\(opencvR[i])")
        }
    }

    // Test 2: Rodrigues small-angle doesn't NaN
    func testRodriguesSmallAngle() {
        let (R, dR) = rodrigues([1e-12, 0, 0])
        XCTAssertFalse(R.contains(where: { $0.isNaN }), "R contains NaN")
        XCTAssertFalse(dR.contains(where: { $0.isNaN }), "dR contains NaN")
        XCTAssertEqual(R[0], 1.0, accuracy: 1e-8)
        XCTAssertEqual(R[4], 1.0, accuracy: 1e-8)
        XCTAssertEqual(R[8], 1.0, accuracy: 1e-8)
    }

    // Test 3: Rodrigues Jacobian vs finite differences
    func testRodriguesJacobian() {
        let rvec = [0.1, 0.2, 0.3]
        let (_, dR_analytical) = rodrigues(rvec)
        let h = 1e-7
        let (R0, _) = rodrigues(rvec)
        for m in 0..<3 {
            var rp = rvec
            rp[m] += h
            let (Rp, _) = rodrigues(rp)
            for i in 0..<9 {
                let fd = (Rp[i] - R0[i]) / h
                let an = dR_analytical[i * 3 + m]
                XCTAssertEqual(an, fd, accuracy: 1e-5,
                    "dR[\(i)]/dr[\(m)]: analytical=\(an) fd=\(fd)")
            }
        }
    }

    // Test 4: Pure projection matches OpenCV
    func testProjectionMatchesOpenCV() {
        let rvec = [-0.057, 0.071, 0.011]
        let tvec = [-0.605, -0.958, 1.218]
        let f = 1.2

        var points3D: [Double] = []
        for i in 0..<20 {
            let x = Double(i) * 0.05
            let y = Double(i % 5) * 0.1
            let z = 0.01 * x * x
            points3D.append(contentsOf: [x, y, z])
        }

        let (projected, _, _, _) = projectAndDifferentiate(
            points3D: points3D, rvec: rvec, tvec: tvec, focalLength: f)

        let pts3D = points3D.map { NSNumber(value: $0) }
        let opencvPts = OpenCVWrapper.projectPointsWith3DPoints(
            pts3D,
            rvec: rvec.map { NSNumber(value: $0) },
            tvec: tvec.map { NSNumber(value: $0) },
            cameraMatrix: [f, 0, 0, 0, f, 0, 0, 0, 1].map { NSNumber(value: $0) },
            distCoeffs: [0, 0, 0, 0, 0].map { NSNumber(value: $0) })

        for i in 0..<20 {
            let pt = opencvPts[i].cgPointValue
            XCTAssertEqual(projected[i*2], Double(pt.x), accuracy: 1e-8,
                "Point \(i) u mismatch")
            XCTAssertEqual(projected[i*2+1], Double(pt.y), accuracy: 1e-8,
                "Point \(i) v mismatch")
        }
    }

    // Test 5: Projection Jacobians vs finite differences
    func testProjectionJacobians() {
        let rvec = [-0.057, 0.071, 0.011]
        let tvec = [-0.605, -0.958, 1.218]
        let f = 1.2
        let points3D = [0.3, 0.2, 0.01]
        let h = 1e-7

        let (p0, _, dR0, dT0) = projectAndDifferentiate(
            points3D: points3D, rvec: rvec, tvec: tvec, focalLength: f)

        // Check dProj/dRvec
        for m in 0..<3 {
            var rp = rvec; rp[m] += h
            let (pp, _, _, _) = projectAndDifferentiate(
                points3D: points3D, rvec: rp, tvec: tvec, focalLength: f)
            let du_fd = (pp[0] - p0[0]) / h
            let dv_fd = (pp[1] - p0[1]) / h
            XCTAssertEqual(dR0[m], du_fd, accuracy: 1e-4,
                "du/dr[\(m)]: analytical=\(dR0[m]) fd=\(du_fd)")
            XCTAssertEqual(dR0[3+m], dv_fd, accuracy: 1e-4,
                "dv/dr[\(m)]: analytical=\(dR0[3+m]) fd=\(dv_fd)")
        }

        // Check dProj/dTvec
        for m in 0..<3 {
            var tp = tvec; tp[m] += h
            let (pp, _, _, _) = projectAndDifferentiate(
                points3D: points3D, rvec: rvec, tvec: tp, focalLength: f)
            let du_fd = (pp[0] - p0[0]) / h
            let dv_fd = (pp[1] - p0[1]) / h
            XCTAssertEqual(dT0[m], du_fd, accuracy: 1e-4,
                "du/dt[\(m)]: analytical=\(dT0[m]) fd=\(du_fd)")
            XCTAssertEqual(dT0[3+m], dv_fd, accuracy: 1e-4,
                "dv/dt[\(m)]: analytical=\(dT0[3+m]) fd=\(dv_fd)")
        }

        // Check dProj/dPoint
        let (_, dP0, _, _) = projectAndDifferentiate(
            points3D: points3D, rvec: rvec, tvec: tvec, focalLength: f)
        for m in 0..<3 {
            var pts = points3D; pts[m] += h
            let (pp, _, _, _) = projectAndDifferentiate(
                points3D: pts, rvec: rvec, tvec: tvec, focalLength: f)
            let du_fd = (pp[0] - p0[0]) / h
            let dv_fd = (pp[1] - p0[1]) / h
            XCTAssertEqual(dP0[m], du_fd, accuracy: 1e-4,
                "du/dP[\(m)]: analytical=\(dP0[m]) fd=\(du_fd)")
            XCTAssertEqual(dP0[3+m], dv_fd, accuracy: 1e-4,
                "dv/dP[\(m)]: analytical=\(dP0[3+m]) fd=\(dv_fd)")
        }
    }
}
