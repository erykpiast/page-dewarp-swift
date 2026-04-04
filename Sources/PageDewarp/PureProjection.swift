// PureProjection.swift
// Pure-Swift Rodrigues rotation + pinhole projection with analytical Jacobians.
// Replaces OpenCV's cv::projectPoints for the optimization hot loop.
//
// Reference: OpenCV cvRodrigues2 in modules/calib3d/src/calibration.cpp

import Foundation

// MARK: - Rodrigues Rotation

/// Convert a rotation vector to a 3×3 rotation matrix and its 9×3 Jacobian.
///
/// Matches OpenCV's `cv::Rodrigues` exactly, including the small-angle case.
///
/// - Parameter rvec: 3-element rotation vector [r1, r2, r3].
/// - Returns:
///   - R: 9-element flat array (row-major 3×3 rotation matrix).
///   - dR_dr: 27-element flat array (9×3 Jacobian, row-major: dR_dr[i*3+m] = ∂R[i]/∂r[m]).
func rodrigues(_ rvec: [Double]) -> (R: [Double], dR_dr: [Double]) {
    let r1 = rvec[0], r2 = rvec[1], r3 = rvec[2]
    let theta2 = r1*r1 + r2*r2 + r3*r3
    let theta = sqrt(theta2)

    // Small angle: R ≈ I + [r]×
    if theta < 1e-10 {
        let R: [Double] = [
            1,   -r3,  r2,
            r3,   1,  -r1,
           -r2,   r1,  1
        ]
        // dR/dr for small angle: derivative of skew-symmetric mapping
        // ∂R/∂r1 = [0,0,0; 0,0,-1; 0,1,0]
        // ∂R/∂r2 = [0,0,1; 0,0,0; -1,0,0]
        // ∂R/∂r3 = [0,-1,0; 1,0,0; 0,0,0]
        var dR = [Double](repeating: 0.0, count: 27)
        // Row-major 9×3: dR[i*3+m] = ∂R_flat[i] / ∂r[m]
        dR[5*3+0] = -1  // ∂R[1,2]/∂r1 = -1
        dR[7*3+0] =  1  // ∂R[2,1]/∂r1 = 1
        dR[2*3+1] =  1  // ∂R[0,2]/∂r2 = 1
        dR[6*3+1] = -1  // ∂R[2,0]/∂r2 = -1
        dR[1*3+2] = -1  // ∂R[0,1]/∂r3 = -1
        dR[3*3+2] =  1  // ∂R[1,0]/∂r3 = 1
        return (R, dR)
    }

    let c = cos(theta), s = sin(theta)
    let invTheta = 1.0 / theta
    let kx = r1 * invTheta, ky = r2 * invTheta, kz = r3 * invTheta
    let v = 1.0 - c // versine

    // Rotation matrix R = cos(θ)·I + (1-cos(θ))·k·kᵀ + sin(θ)·[k]×
    let R: [Double] = [
        c + kx*kx*v,       kx*ky*v - kz*s,    kx*kz*v + ky*s,
        ky*kx*v + kz*s,    c + ky*ky*v,        ky*kz*v - kx*s,
        kz*kx*v - ky*s,    kz*ky*v + kx*s,     c + kz*kz*v
    ]

    // Jacobian dR/dr: 9×3 matrix
    // For each R element (i,j) and each r component m:
    // ∂R_ij/∂r_m = ∂θ/∂r_m · dR_ij/dθ + Σ_n ∂k_n/∂r_m · ∂R_ij/∂k_n
    //
    // where ∂θ/∂r_m = k_m, ∂k_n/∂r_m = (δ_nm - k_n·k_m)/θ
    //
    // dR_ij/dθ = -sin(θ)·δ_ij + sin(θ)·k_i·k_j + cos(θ)·[k]×_ij
    // ∂R_ij/∂k_n involves the derivatives of k·kᵀ and [k]× w.r.t. k_n

    let k = [kx, ky, kz]
    var dR = [Double](repeating: 0.0, count: 27)

    // Precompute the skew-symmetric matrix [k]× as flat array
    let skew: [Double] = [
         0,  -kz,  ky,
         kz,  0,  -kx,
        -ky,  kx,  0
    ]

    for i in 0..<3 {
        for j in 0..<3 {
            let flatIdx = i * 3 + j
            let dij: Double = (i == j) ? 1.0 : 0.0

            // dR_ij/dθ
            let dR_dtheta = -s * dij + s * k[i] * k[j] + c * skew[flatIdx]

            for m in 0..<3 {
                // ∂θ/∂r_m = k_m
                var val = k[m] * dR_dtheta

                // ∂k_n/∂r_m contributions
                for n in 0..<3 {
                    let dk_n_dr_m = ((n == m) ? 1.0 : 0.0) - k[n] * k[m]
                    let dk_n_dr_m_scaled = dk_n_dr_m * invTheta

                    // ∂(k_i·k_j)/∂k_n = δ_in·k_j + k_i·δ_jn
                    let d_kkT = ((i == n) ? k[j] : 0.0) + ((j == n) ? k[i] : 0.0)

                    // ∂[k]×_ij/∂k_n
                    var d_skew = 0.0
                    if i == 0 && j == 1 { d_skew = (n == 2) ? -1.0 : 0.0 }      // -kz
                    else if i == 0 && j == 2 { d_skew = (n == 1) ? 1.0 : 0.0 }   //  ky
                    else if i == 1 && j == 0 { d_skew = (n == 2) ? 1.0 : 0.0 }   //  kz
                    else if i == 1 && j == 2 { d_skew = (n == 0) ? -1.0 : 0.0 }  // -kx
                    else if i == 2 && j == 0 { d_skew = (n == 1) ? -1.0 : 0.0 }  // -ky
                    else if i == 2 && j == 1 { d_skew = (n == 0) ? 1.0 : 0.0 }   //  kx

                    val += dk_n_dr_m_scaled * (v * d_kkT + s * d_skew)
                }

                dR[flatIdx * 3 + m] = val
            }
        }
    }

    return (R, dR)
}

// MARK: - Projection with Jacobians

/// Project N 3D points through the pinhole camera model with analytical Jacobians.
///
/// No distortion, no principal point offset. Camera matrix K = diag(f, f, 1).
///
/// - Parameters:
///   - points3D: Flat array [x0,y0,z0, x1,y1,z1, ...] of N points.
///   - rvec: 3-element rotation vector.
///   - tvec: 3-element translation vector.
///   - focalLength: Scalar focal length.
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
) -> (projected: [Double], dProj_dPoint: [Double], dProj_dRvec: [Double], dProj_dTvec: [Double]) {
    let (R, dR_dr) = rodrigues(rvec)
    let tx = tvec[0], ty = tvec[1], tz = tvec[2]
    let nPoints = points3D.count / 3

    var projected = [Double](repeating: 0.0, count: nPoints * 2)
    var dProj_dPoint = [Double](repeating: 0.0, count: nPoints * 6)
    var dProj_dRvec = [Double](repeating: 0.0, count: nPoints * 6)
    var dProj_dTvec = [Double](repeating: 0.0, count: nPoints * 6)

    for p in 0..<nPoints {
        let X = points3D[p*3], Y = points3D[p*3+1], Z = points3D[p*3+2]

        // Camera space: P_cam = R·P + t
        let cx = R[0]*X + R[1]*Y + R[2]*Z + tx
        let cy = R[3]*X + R[4]*Y + R[5]*Z + ty
        let cz = R[6]*X + R[7]*Y + R[8]*Z + tz

        // Perspective division + focal length
        let iz = 1.0 / cz
        let u = focalLength * cx * iz
        let v = focalLength * cy * iz

        projected[p*2] = u
        projected[p*2+1] = v

        // Perspective Jacobian: d(u,v)/d(cx,cy,cz)
        let f_iz = focalLength * iz
        let f_iz2 = focalLength * iz * iz
        // J = [[f/cz, 0, -f*cx/cz²], [0, f/cz, -f*cy/cz²]]
        let J00 = f_iz, J02 = -f_iz2 * cx
        let J11 = f_iz, J12 = -f_iz2 * cy

        // dProj/dPoint: J_persp · R
        // du/dX = J00·R00 + J02·R20, du/dY = J00·R01 + J02·R21, du/dZ = J00·R02 + J02·R22
        // dv/dX = J11·R10 + J12·R20, dv/dY = J11·R11 + J12·R21, dv/dZ = J11·R12 + J12·R22
        let off = p * 6
        dProj_dPoint[off+0] = J00*R[0] + J02*R[6]  // du/dX
        dProj_dPoint[off+1] = J00*R[1] + J02*R[7]  // du/dY
        dProj_dPoint[off+2] = J00*R[2] + J02*R[8]  // du/dZ
        dProj_dPoint[off+3] = J11*R[3] + J12*R[6]  // dv/dX
        dProj_dPoint[off+4] = J11*R[4] + J12*R[7]  // dv/dY
        dProj_dPoint[off+5] = J11*R[5] + J12*R[8]  // dv/dZ

        // dProj/dRvec: J_persp · (dR/dr_m · P) for each m
        for m in 0..<3 {
            // d(cx)/dr_m = dR[0,m]*X + dR[1,m]*Y + dR[2,m]*Z (where dR is 9×3)
            let dcx = dR_dr[0*3+m]*X + dR_dr[1*3+m]*Y + dR_dr[2*3+m]*Z
            let dcy = dR_dr[3*3+m]*X + dR_dr[4*3+m]*Y + dR_dr[5*3+m]*Z
            let dcz = dR_dr[6*3+m]*X + dR_dr[7*3+m]*Y + dR_dr[8*3+m]*Z
            dProj_dRvec[off+m]   = J00*dcx + J02*dcz  // du/dr_m
            dProj_dRvec[off+3+m] = J11*dcy + J12*dcz  // dv/dr_m
        }

        // dProj/dTvec: J_persp · I
        dProj_dTvec[off+0] = J00   // du/dtx
        dProj_dTvec[off+1] = 0     // du/dty
        dProj_dTvec[off+2] = J02   // du/dtz
        dProj_dTvec[off+3] = 0     // dv/dtx
        dProj_dTvec[off+4] = J11   // dv/dty
        dProj_dTvec[off+5] = J12   // dv/dtz
    }

    return (projected, dProj_dPoint, dProj_dRvec, dProj_dTvec)
}
