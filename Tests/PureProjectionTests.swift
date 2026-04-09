// PureProjectionTests.swift

import XCTest
@testable import PageDewarp

final class PureProjectionTests: XCTestCase {

    // Test 1: Rodrigues rotation matches Python/OpenCV golden values
    func testRodriguesMatchesOpenCV() {
        let rvec = [0.1, 0.2, 0.3]
        let (R, _) = rodrigues(rvec)
        // Golden rotation matrix from Python cv2.Rodrigues([0.1, 0.2, 0.3])
        let goldenR: [Double] = [
            0.9357548032779188, -0.28316496056507373, 0.21019170595074288,
            0.3029327134026371, 0.9505806179060914, -0.06803131640494002,
            -0.18054007669439776, 0.12733457491763028, 0.9752903089530457,
        ]
        for i in 0..<9 {
            XCTAssertEqual(R[i], goldenR[i], accuracy: 1e-10,
                "R[\(i)] mismatch: swift=\(R[i]) golden=\(goldenR[i])")
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

    // Test 4: Pure projection matches Python/OpenCV golden values
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

        // Golden projected points from Python cv2.projectPoints with same inputs
        let goldenPts: [(Double, Double)] = [
            (-0.5960591133004925, -0.9438423645320195),
            (-0.5523719708449378, -0.8514667754879058),
            (-0.5079909492720853, -0.7576417284213769),
            (-0.4629056381484923, -0.6623456168702244),
            (-0.4171054839101698, -0.5655565396713497),
            (-0.35535815979765084, -0.9550717478017853),
            (-0.309102711209272, -0.8612565011907795),
            (-0.2621335165528024, -0.7659881267735411),
            (-0.21444033907972365, -0.6692457379590012),
            (-0.1660128161530344, -0.5710081904871459),
            (-0.10746696523696844, -0.9655390523647291),
            (-0.05866970793273751, -0.8703331266864773),
            (-0.009141941504840066, -0.7736738371876267),
            (0.04112629015793571, -0.6755411340045706),
            (0.0921450491409233, -0.5759147492498851),
            (0.14711712791907805, -0.9751797385593881),
            (0.19841221584268343, -0.8786385877320769),
            (0.25045109453415265, -0.7806474430762282),
            (0.30324332927700925, -0.6811872064338492),
            (0.3567985689881251, -0.5802386034470306),
        ]

        for i in 0..<20 {
            XCTAssertEqual(projected[i*2], goldenPts[i].0, accuracy: 1e-8,
                "Point \(i) u mismatch")
            XCTAssertEqual(projected[i*2+1], goldenPts[i].1, accuracy: 1e-8,
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
