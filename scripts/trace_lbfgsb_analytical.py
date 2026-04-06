#!/usr/bin/env python3
"""Test Python L-BFGS-B with analytical gradient to diagnose Swift divergence."""

import json
import os
import sys
import math
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, '/opt/homebrew/lib/python3.14/site-packages')

from scipy.optimize import minimize

from page_dewarp.options import cfg, Config
from page_dewarp.keypoints import make_keypoint_index, project_keypoints
from page_dewarp.optimise._base import make_objective
from page_dewarp.solve import get_default_params
from page_dewarp.mask import Mask
from page_dewarp.contours import get_contours
from page_dewarp.spans import assemble_spans, keypoints_from_samples, sample_spans

DEBUG_DIR = Path.home() / 'Desktop' / 'lbfgsb-debug'
DEBUG_DIR.mkdir(exist_ok=True)

IMG_PATH = str(Path.home() / 'Desktop' / 'IMG_1389.jpeg')

config = Config()
cv2_img = cv2.imread(IMG_PATH)
h, w = cv2_img.shape[:2]
scl_x = w / config.SCREEN_MAX_W
scl_y = h / config.SCREEN_MAX_H
scl = max(scl_x, scl_y)
if scl > 1:
    scl = math.ceil(scl)
    new_w = int(w / scl)
    new_h = int(h / scl)
    small = cv2.resize(cv2_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
else:
    small = cv2_img.copy()

s_h, s_w = small.shape[:2]
pagemask = np.zeros((s_h, s_w), dtype=np.uint8)
xmin, ymin = config.PAGE_MARGIN_X, config.PAGE_MARGIN_Y
xmax, ymax = s_w - xmin, s_h - ymin
cv2.rectangle(pagemask, (xmin, ymin), (xmax, ymax), 255, -1)
page_outline = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]], dtype=np.float32)

name = "IMG_1389"
mask_text = Mask(name, small, pagemask, text=True)
contour_list = get_contours(name, small, mask_text.value)
spans = assemble_spans(name, small, pagemask, contour_list)
span_points = sample_spans(small.shape, spans)
corners, ycoords, xcoords = keypoints_from_samples(name, small, pagemask, page_outline, span_points)
rough_dims, span_counts, params0 = get_default_params(corners, ycoords, xcoords)
dstpoints = np.vstack((corners[0].reshape((1, 1, 2)),) + tuple(span_points))
keypoint_index = make_keypoint_index(span_counts)
objective_fn = make_objective(dstpoints, keypoint_index, cfg.SHEAR_COST, slice(*cfg.RVEC_IDX))

print(f"params shape: {params0.shape}")
print(f"Initial rvec: {params0[0:3]}")
print(f"Initial objective: {objective_fn(params0):.6f}")

# Implement analytical gradient matching Swift's AnalyticalGradient.swift
import cv2 as cv  # for Rodrigues

FOCAL_LENGTH = 1.2

def rodrigues_with_jac(rvec):
    """Returns (R 3x3, dR 9x3) where dR[i*3+j] = dR.flat[i]/dr[j]"""
    rvec_col = np.array(rvec, dtype=np.float64).reshape(3, 1)
    R_col, jac = cv.Rodrigues(rvec_col)
    R = R_col.flatten()  # length 9
    # jac shape is (3, 9) from Rodrigues - rows are dr, cols are dR elements
    # We want dR[i*3+m] = d(R.flat[i])/d(r[m])
    # cv.Rodrigues jac: jac[m, i] = d(R.flat[i])/d(r[m])
    # So dR_dr[i*3+m] = jac[m, i]
    dR = np.zeros(9*3)
    for i in range(9):
        for m in range(3):
            dR[i*3+m] = jac[m, i]
    return R, dR

def analytical_gradient(pvec, dstpoints_2d, keypoint_index, shear_cost=0.0, focal_length=FOCAL_LENGTH):
    """
    Compute (f, grad) analytically, matching Swift's objectiveAndGradient.
    dstpoints_2d: shape (N, 2) float array
    """
    n = len(pvec)
    grad = np.zeros(n)
    f = 0.0

    rvec = pvec[0:3]
    tvec = pvec[3:6]
    raw_alpha = pvec[6]
    raw_beta = pvec[7]
    alpha = float(np.clip(raw_alpha, -0.5, 0.5))
    beta = float(np.clip(raw_beta, -0.5, 0.5))
    alpha_clamped = abs(raw_alpha) >= 0.5
    beta_clamped = abs(raw_beta) >= 0.5

    cubic_a = alpha + beta
    cubic_b = -2*alpha - beta
    cubic_c = alpha

    n_pts = len(keypoint_index)
    R, dR_dr = rodrigues_with_jac(rvec)
    tx, ty, tz = tvec[0], tvec[1], tvec[2]

    for k in range(n_pts):
        if k == 0:
            x = 0.0; y = 0.0
        else:
            x = pvec[keypoint_index[k][0]]
            y = pvec[keypoint_index[k][1]]
        z = ((cubic_a * x + cubic_b) * x + cubic_c) * x

        cx = R[0]*x + R[1]*y + R[2]*z + tx
        cy = R[3]*x + R[4]*y + R[5]*z + ty
        cz = R[6]*x + R[7]*y + R[8]*z + tz

        iz = 1.0 / cz
        u = focal_length * cx * iz
        v = focal_length * cy * iz

        dst = dstpoints_2d[k]
        du = u - dst[0]
        dv = v - dst[1]
        f += du*du + dv*dv

        eu = 2.0 * du
        ev = 2.0 * dv

        f_iz = focal_length * iz
        f_iz2 = focal_length * iz * iz
        J00 = f_iz
        J02 = -f_iz2 * cx
        J11 = f_iz
        J12 = -f_iz2 * cy

        # rvec gradient
        for m in range(3):
            dcx = dR_dr[0*3+m]*x + dR_dr[1*3+m]*y + dR_dr[2*3+m]*z
            dcy = dR_dr[3*3+m]*x + dR_dr[4*3+m]*y + dR_dr[5*3+m]*z
            dcz = dR_dr[6*3+m]*x + dR_dr[7*3+m]*y + dR_dr[8*3+m]*z
            du_dr = J00*dcx + J02*dcz
            dv_dr = J11*dcy + J12*dcz
            grad[m] += eu*du_dr + ev*dv_dr

        # tvec gradient
        grad[3] += eu * J00
        grad[4] += ev * J11
        grad[5] += eu * J02 + ev * J12

        # cubic alpha, beta
        du_dz = J00*R[2] + J02*R[8]
        dv_dz = J11*R[5] + J12*R[8]
        err_dot_dz = eu*du_dz + ev*dv_dz

        x2 = x*x; x3 = x2*x
        if not alpha_clamped:
            dz_dalpha = x3 - 2*x2 + x
            grad[6] += err_dot_dz * dz_dalpha
        if not beta_clamped:
            dz_dbeta = x3 - x2
            grad[7] += err_dot_dz * dz_dbeta

        if k == 0:
            continue

        x_idx = keypoint_index[k][0]
        y_idx = keypoint_index[k][1]

        du_dy = J00*R[1] + J02*R[7]
        dv_dy = J11*R[4] + J12*R[7]
        grad[y_idx] += eu*du_dy + ev*dv_dy

        dz_dx = 3*cubic_a*x2 - 2*(2*alpha + beta)*x + alpha
        du_dx = J00*(R[0] + R[2]*dz_dx) + J02*(R[6] + R[8]*dz_dx)
        dv_dx = J11*(R[3] + R[5]*dz_dx) + J12*(R[6] + R[8]*dz_dx)
        grad[x_idx] += eu*du_dx + ev*dv_dx

    if shear_cost > 0:
        f += shear_cost * pvec[0]*pvec[0]
        grad[0] += 2.0 * shear_cost * pvec[0]

    return f, grad

# Flatten dstpoints for analytical gradient
dstpoints_flat = dstpoints.reshape(-1, 2)

# Verify analytical gradient matches finite differences
print("\n--- Verifying analytical gradient at initial params ---")
f_anal, g_anal = analytical_gradient(params0, dstpoints_flat, keypoint_index)
print(f"Analytical f: {f_anal:.6f}")
f_check = objective_fn(params0)
print(f"Objective f:  {f_check:.6f}")
print(f"Match: {abs(f_anal - f_check) < 1e-10}")

# Finite diff gradient check
eps = 1e-6
g_fd = np.zeros(len(params0))
for i in range(min(10, len(params0))):
    p_plus = params0.copy(); p_plus[i] += eps
    p_minus = params0.copy(); p_minus[i] -= eps
    g_fd[i] = (objective_fn(p_plus) - objective_fn(p_minus)) / (2*eps)
print(f"\nGradient check (first 10 params):")
for i in range(10):
    print(f"  [{i}] analytical={g_anal[i]:.8f}  fd={g_fd[i]:.8f}  diff={abs(g_anal[i]-g_fd[i]):.2e}")

def make_analytical_obj(dstpoints_2d, keypoint_index):
    def obj_and_grad(pvec):
        return analytical_gradient(pvec, dstpoints_2d, keypoint_index, cfg.SHEAR_COST)
    return obj_and_grad

obj_and_grad = make_analytical_obj(dstpoints_flat, keypoint_index)

def run_lbfgsb_with_jac(params, maxcor, label, use_analytical_grad):
    iterations = []
    iter_count = [0]

    def callback(xk):
        iter_count[0] += 1
        f = objective_fn(xk)
        if iter_count[0] <= 5 or iter_count[0] % 100 == 0:
            print(f"  [{label}] iter {iter_count[0]:4d}: f={f:.6f} rvec[0]={xk[0]:.6f}")
        iterations.append({'iter': iter_count[0], 'f': float(f), 'rvec': xk[0:3].tolist()})

    if use_analytical_grad:
        result = minimize(
            obj_and_grad,
            params.copy(),
            method='L-BFGS-B',
            jac=True,  # function returns (f, grad)
            options={'maxiter': 600000, 'maxcor': maxcor},
            callback=callback,
        )
    else:
        result = minimize(
            objective_fn,
            params.copy(),
            method='L-BFGS-B',
            options={'maxiter': 600000, 'maxcor': maxcor},
            callback=callback,
        )
    print(f"[{label}] Done: {result.nfev} evals, f={result.fun:.6f}, rvec[0]={result.x[0]:.6f}")
    return result, iterations

print("\n--- 1. Finite diff, maxcor=10 (Python default) ---")
r1, i1 = run_lbfgsb_with_jac(params0, 10, 'FD,mcor=10', False)

print("\n--- 2. Finite diff, maxcor=100 (Swift config) ---")
r2, i2 = run_lbfgsb_with_jac(params0, 100, 'FD,mcor=100', False)

print("\n--- 3. Analytical grad, maxcor=10 ---")
r3, i3 = run_lbfgsb_with_jac(params0, 10, 'AG,mcor=10', True)

print("\n--- 4. Analytical grad, maxcor=100 (Swift equivalent) ---")
r4, i4 = run_lbfgsb_with_jac(params0, 100, 'AG,mcor=100', True)

print("\n=== SUMMARY ===")
print(f"Initial rvec[0]:                        {params0[0]:.6f}")
print(f"1. FD + maxcor=10 (Python default):     {r1.x[0]:.6f}  (expected ≈0.053)")
print(f"2. FD + maxcor=100:                     {r2.x[0]:.6f}")
print(f"3. Analytical + maxcor=10:              {r3.x[0]:.6f}")
print(f"4. Analytical + maxcor=100 (Swift):     {r4.x[0]:.6f}  (expected ≈0.184)")
print(f"")
print(f"Objective values:")
print(f"1. FD + maxcor=10:    f={r1.fun:.6f}")
print(f"2. FD + maxcor=100:   f={r2.fun:.6f}")
print(f"3. Analytical + mc10: f={r3.fun:.6f}")
print(f"4. Analytical + mc100: f={r4.fun:.6f}")

trace_data = {
    'initial_params': params0.tolist(),
    'initial_f': float(f_check),
    'finite_diff_maxcor10': {'rvec': r1.x[0:3].tolist(), 'f': float(r1.fun), 'nfev': r1.nfev},
    'finite_diff_maxcor100': {'rvec': r2.x[0:3].tolist(), 'f': float(r2.fun), 'nfev': r2.nfev},
    'analytical_maxcor10': {'rvec': r3.x[0:3].tolist(), 'f': float(r3.fun), 'nfev': r3.nfev},
    'analytical_maxcor100': {'rvec': r4.x[0:3].tolist(), 'f': float(r4.fun), 'nfev': r4.nfev},
}
with open(DEBUG_DIR / 'analytical_gradient_comparison.json', 'w') as f:
    json.dump(trace_data, f, indent=2)
print(f"\nSaved to {DEBUG_DIR}/analytical_gradient_comparison.json")
