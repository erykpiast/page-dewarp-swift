// SolvePnPDLT.swift
// Pure-Swift DLT homography + decomposition for the planar solvePnP case.
// Replaces OpenCV's cv::solvePnP for 4 coplanar (Z=0) point correspondences.
//
// Reference: Hartley & Zisserman "Multiple View Geometry", Chapter 4 (DLT)
// Reference: OpenCV calib3d module decompose homography

import Foundation
import Accelerate

// MARK: - Public API

/// Solve the PnP problem for a planar target (all Z=0) using DLT homography decomposition.
///
/// Builds a 2n×9 DLT matrix, decomposes via SVD to recover the homography,
/// then recovers rotation and translation from K^{-1}·H. Projects R onto SO(3)
/// via a second SVD.
///
/// - Parameters:
///   - objectPoints: Flat [X0,Y0,Z0, X1,Y1,Z1, ...] of N 3D points (Z must be 0).
///   - imagePoints: Flat [u0,v0, u1,v1, ...] of N 2D image points.
///   - cameraMatrix: Flat row-major 3×3 camera intrinsics K.
/// - Returns: `(rvec, tvec)` or `nil` for degenerate inputs.
func solvePnPPlanar(
    objectPoints: [Double],
    imagePoints: [Double],
    cameraMatrix: [Double]
) -> (rvec: [Double], tvec: [Double])? {
    let n = objectPoints.count / 3
    guard n >= 4, imagePoints.count == n * 2 else { return nil }

    // Build 2n×9 DLT matrix A in row-major format.
    // Each 3D-2D correspondence (X,Y,0) ↔ (u,v) gives two rows:
    //   [X Y 1  0 0 0  -uX -uY -u]
    //   [0 0 0  X Y 1  -vX -vY -v]
    // Ported from Hartley & Zisserman Algorithm 4.1
    var A = [Double](repeating: 0, count: 2 * n * 9)
    for i in 0..<n {
        let X = objectPoints[i*3], Y = objectPoints[i*3+1]
        // Z is ignored — planar target has Z=0
        let u = imagePoints[i*2], v = imagePoints[i*2+1]
        let r0 = 2*i, r1 = 2*i + 1
        // Row r0: [X Y 1  0 0 0  -uX -uY -u]
        A[r0*9 + 0] = X;   A[r0*9 + 1] = Y;   A[r0*9 + 2] = 1
        A[r0*9 + 6] = -u*X; A[r0*9 + 7] = -u*Y; A[r0*9 + 8] = -u
        // Row r1: [0 0 0  X Y 1  -vX -vY -v]
        A[r1*9 + 3] = X;   A[r1*9 + 4] = Y;   A[r1*9 + 5] = 1
        A[r1*9 + 6] = -v*X; A[r1*9 + 7] = -v*Y; A[r1*9 + 8] = -v
    }

    // SVD of A to find the homography h = last right singular vector of A.
    // LAPACK uses column-major storage. To compute SVD of our row-major A (2n×9)
    // we pass it transposed (as a column-major 2n×9) to LAPACK's dgesdd_,
    // treating our memory as a (9×2n) column-major matrix.
    // Then: SVD of A^T (9×2n) = U_l * S * VT_l
    //       ↔ A = V_l * S * U_l^T
    // The right singular vector of A for smallest singular value = last column of V_l
    //   = last column of U_l (from SVD of A^T) = last column of U_l
    //
    // LAPACK dgesdd_ for A^T (m_l=9, n_l=2n):
    //   s has min(9, 2n) = 9 values (for n>=5) or 2n values (n=4, so 8)
    // We need jobz='A' to get all 9 columns of U_l, but if n=4 (2n=8 < 9)
    // we only get 8 singular values and U_l is 9×8 in economy mode 'S'.
    // Use 'A' to guarantee full U_l (9×9), from which we take the last column.

    let m_l = 9       // rows of A^T
    let n_l = 2 * n   // cols of A^T

    var jobz = Int8(UInt8(ascii: "A"))  // compute all singular vectors
    var m_lapack = __CLPK_integer(m_l)
    var n_lapack = __CLPK_integer(n_l)
    var lda = __CLPK_integer(m_l)       // leading dimension of A (column-major)
    var ldu = __CLPK_integer(m_l)       // leading dimension of U
    var ldvt = __CLPK_integer(n_l)      // leading dimension of VT

    var s = [Double](repeating: 0, count: min(m_l, n_l))
    var U = [Double](repeating: 0, count: m_l * m_l)   // 9×9
    var VT = [Double](repeating: 0, count: n_l * n_l)  // (2n)×(2n)

    var iwork = [__CLPK_integer](repeating: 0, count: 8 * min(m_l, n_l))
    var info = __CLPK_integer(0)

    // Query optimal workspace size
    var lwork = __CLPK_integer(-1)
    var workQuery = [Double](repeating: 0, count: 1)
    dgesdd_(&jobz, &m_lapack, &n_lapack, &A, &lda, &s, &U, &ldu, &VT, &ldvt,
            &workQuery, &lwork, &iwork, &info)
    lwork = __CLPK_integer(workQuery[0])
    var work = [Double](repeating: 0, count: Int(lwork))

    // Actual SVD call
    dgesdd_(&jobz, &m_lapack, &n_lapack, &A, &lda, &s, &U, &ldu, &VT, &ldvt,
            &work, &lwork, &iwork, &info)
    guard info == 0 else { return nil }

    // Extract h = last column of U_l (9×9, column-major): column index 8
    // Column-major: U[row + col*m_l]  →  last col: U[row + 8*9] for row in 0..<9
    // For n=4 (2n=8 < 9): singular values 0-7 are real, value 8 = 0 → null vector
    let h = (0..<9).map { U[$0 + 8 * m_l] }

    // Reshape h into 3×3 homography H (row-major)
    let H = h  // H[i*3+j] = h[i*3+j]

    // Decompose: M = K^{-1} · H
    // K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] (row-major, indices 0..8)
    let fx = cameraMatrix[0], fy = cameraMatrix[4]
    let cx = cameraMatrix[2], cy = cameraMatrix[5]
    guard fx != 0, fy != 0 else { return nil }
    let ifx = 1.0 / fx, ify = 1.0 / fy

    // K^{-1} · H (3×3 × 3×3 = 3×3, row-major)
    // K^{-1} = [[1/fx, 0, -cx/fx], [0, 1/fy, -cy/fy], [0, 0, 1]]
    var M = [Double](repeating: 0, count: 9)
    for col in 0..<3 {
        M[0*3+col] = ifx * H[0*3+col] - (cx * ifx) * H[2*3+col]
        M[1*3+col] = ify * H[1*3+col] - (cy * ify) * H[2*3+col]
        M[2*3+col] = H[2*3+col]
    }

    // Normalize by scale = ||col0(M)||
    let scale = sqrt(M[0]*M[0] + M[3]*M[3] + M[6]*M[6])
    guard scale > 1e-10 else { return nil }
    let inv_scale = 1.0 / scale
    for i in 0..<9 { M[i] *= inv_scale }

    // M = [r1 | r2 | t] in column representation (each column is a 3-vector)
    // r1 = M[:,0], r2 = M[:,1], t = M[:,2]
    let r1 = [M[0], M[3], M[6]]
    let r2 = [M[1], M[4], M[7]]
    let tvec = [M[2], M[5], M[8]]

    // r3 = r1 × r2
    let r3 = cross3(r1, r2)

    // Assemble approximate rotation matrix R_approx = [r1 | r2 | r3] (column-major for SVD)
    // We'll pass it in row-major order but tell LAPACK it's column-major (effectively transposing)
    // R_approx (row-major): R[row*3+col]
    var R_approx = [Double](repeating: 0, count: 9)
    R_approx[0] = r1[0]; R_approx[1] = r2[0]; R_approx[2] = r3[0]
    R_approx[3] = r1[1]; R_approx[4] = r2[1]; R_approx[5] = r3[1]
    R_approx[6] = r1[2]; R_approx[7] = r2[2]; R_approx[8] = r3[2]

    // Project R_approx onto SO(3) via SVD: R = U2 · V2^T
    // R_approx = U2 · S2 · V2^T → R = U2 · V2^T
    // We pass R_approx (3×3 row-major) as a column-major 3×3 matrix to LAPACK.
    // dgesdd_: m=3, n=3 → s has 3 values, U2 is 3×3, V2T is 3×3
    var jobz2 = Int8(UInt8(ascii: "A"))
    var m3_m = __CLPK_integer(3)
    var m3_n = __CLPK_integer(3)
    var lda3 = __CLPK_integer(3)
    var ldu3 = __CLPK_integer(3)
    var ldvt3 = __CLPK_integer(3)
    var s3 = [Double](repeating: 0, count: 3)
    var U2 = [Double](repeating: 0, count: 9)
    var VT2 = [Double](repeating: 0, count: 9)
    var iwork2 = [__CLPK_integer](repeating: 0, count: 24)
    var info2 = __CLPK_integer(0)

    var lwork2 = __CLPK_integer(-1)
    var workQuery2 = [Double](repeating: 0, count: 1)
    dgesdd_(&jobz2, &m3_m, &m3_n, &R_approx, &lda3, &s3, &U2, &ldu3, &VT2, &ldvt3,
            &workQuery2, &lwork2, &iwork2, &info2)
    lwork2 = __CLPK_integer(workQuery2[0])
    var work2 = [Double](repeating: 0, count: Int(lwork2))
    dgesdd_(&jobz2, &m3_m, &m3_n, &R_approx, &lda3, &s3, &U2, &ldu3, &VT2, &ldvt3,
            &work2, &lwork2, &iwork2, &info2)
    guard info2 == 0 else { return nil }

    // R = U2 · VT2 (both column-major 3×3 from LAPACK)
    // But we passed our row-major R_approx as column-major, so LAPACK computed SVD of R_approx^T
    // R_approx^T = U2 * S2 * VT2
    // R_approx = VT2^T * S2 * U2^T
    // SO(3) projection of R_approx = VT2^T * U2^T = (U2 * VT2)^T
    //
    // We need R = V * U^T (where V = VT2^T, U = U2)
    // But since we're using LAPACK column-major, VT2 from LAPACK is V^T of R_approx^T,
    // which = U^T of R_approx. Similarly U2 = V of R_approx.
    // R_proj = U * V^T = U2^T_colmaj * VT2^T_colmaj ... getting complex.
    //
    // Simpler: compute R = U2 * VT2 in column-major, then interpret result.
    // Use cblas_dgemm for 3×3 matrix multiply.
    var R_proj = [Double](repeating: 0, count: 9)
    // In column-major: C = alpha * A * B + beta * C
    // cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                3, 3, 3, 1.0, U2, 3, VT2, 3, 0.0, &R_proj, 3)

    // Ensure det(R_proj) = +1 (not -1)
    let det = det3x3colmaj(R_proj)
    if det < 0 {
        // Flip sign of last column of U2 and recompute
        var U2_fixed = U2
        U2_fixed[6] = -U2_fixed[6]  // col 2, row 0
        U2_fixed[7] = -U2_fixed[7]  // col 2, row 1
        U2_fixed[8] = -U2_fixed[8]  // col 2, row 2
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    3, 3, 3, 1.0, U2_fixed, 3, VT2, 3, 0.0, &R_proj, 3)
    }

    // R_proj (column-major) = U2 * VT2 = SO(3) projection of R_approx^T (because
    // we passed row-major R_approx to LAPACK which interprets it as column-major R_approx^T).
    //
    // We want SO(3) projection of R_approx = (SO(3) proj of R_approx^T)^T.
    //
    // For a column-major matrix P, P^T in row-major has the same flat storage:
    //   P^T_rowmaj[i*3+j] = P_colmaj[j + i*3] = P_flat[same index rearranged]
    // But more concretely: P_colmaj = [p00,p10,p20, p01,p11,p21, p02,p12,p22]
    //                    P^T_rowmaj = [p00,p10,p20, p01,p11,p21, p02,p12,p22] ← same flat array!
    // So reinterpreting R_proj (col-major) as row-major gives us R_proj^T = SO(3) proj of R_approx.
    let R_rowmaj = R_proj  // col-major P reinterpreted as row-major = P^T = SO(3) proj of R_approx

    // Convert R to rvec
    let rvec = rotationMatrixToRvec(R_rowmaj)

    return (rvec: rvec, tvec: tvec)
}

// MARK: - Inverse Rodrigues

/// Convert a 3×3 rotation matrix (row-major) to a rotation vector.
///
/// Implements the inverse Rodrigues formula with three regimes:
/// near-identity (small angle), near-π, and general case.
///
/// Round-trips with `rodriguesRotationOnly()` to <1e-12 error.
func rotationMatrixToRvec(_ R: [Double]) -> [Double] {
    let trace = R[0] + R[4] + R[8]
    let cosAngle = min(1.0, max(-1.0, (trace - 1.0) / 2.0))
    let angle = acos(cosAngle)

    // Near-identity: small angle approximation
    if angle < 1e-10 {
        return [(R[7]-R[5])/2.0, (R[2]-R[6])/2.0, (R[3]-R[1])/2.0]
    }

    // Near-π: use diagonal to find the dominant axis
    if angle > .pi - 1e-6 {
        let s00 = (R[0] + 1.0) / 2.0
        let s11 = (R[4] + 1.0) / 2.0
        let s22 = (R[8] + 1.0) / 2.0
        var axis: [Double]
        if s00 >= s11 && s00 >= s22 {
            let k0 = sqrt(max(0.0, s00))
            guard k0 > 1e-10 else { return [angle, 0, 0] }
            axis = [k0, (R[1]+R[3])/(4.0*k0), (R[2]+R[6])/(4.0*k0)]
        } else if s11 >= s22 {
            let k1 = sqrt(max(0.0, s11))
            guard k1 > 1e-10 else { return [0, angle, 0] }
            axis = [(R[1]+R[3])/(4.0*k1), k1, (R[5]+R[7])/(4.0*k1)]
        } else {
            let k2 = sqrt(max(0.0, s22))
            guard k2 > 1e-10 else { return [0, 0, angle] }
            axis = [(R[2]+R[6])/(4.0*k2), (R[5]+R[7])/(4.0*k2), k2]
        }
        let norm = sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2])
        guard norm > 1e-10 else { return [0, 0, 0] }
        return axis.map { $0 / norm * angle }
    }

    // General case
    let s = angle / (2.0 * sin(angle))
    return [(R[7] - R[5]) * s, (R[2] - R[6]) * s, (R[3] - R[1]) * s]
}

// MARK: - Private helpers

/// Cross product of two 3-vectors.
private func cross3(_ a: [Double], _ b: [Double]) -> [Double] {
    return [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ]
}

/// Determinant of a 3×3 column-major matrix.
private func det3x3colmaj(_ M: [Double]) -> Double {
    // Column-major: M[row + col*3]
    let a = M[0], b = M[3], c = M[6]
    let d = M[1], e = M[4], f = M[7]
    let g = M[2], h = M[5], i = M[8]
    return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
}

