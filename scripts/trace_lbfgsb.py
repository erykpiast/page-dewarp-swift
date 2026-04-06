#!/usr/bin/env python3
"""Trace L-BFGS-B iterations for IMG_1389.jpeg to compare with Swift."""

import json
import os
import sys
from pathlib import Path
import math

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
print(f"Loading image: {IMG_PATH}")

config = Config()
cv2_img = cv2.imread(IMG_PATH)
if cv2_img is None:
    print(f"ERROR: Could not load image {IMG_PATH}")
    sys.exit(1)
print(f"Loaded image shape: {cv2_img.shape}")

# Resize to screen (same as Python defaults)
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
print(f"small shape: {small.shape}")

# Create page mask
s_h, s_w = small.shape[:2]
pagemask = np.zeros((s_h, s_w), dtype=np.uint8)
xmin = config.PAGE_MARGIN_X
ymin = config.PAGE_MARGIN_Y
xmax = s_w - xmin
ymax = s_h - ymin
cv2.rectangle(pagemask, (xmin, ymin), (xmax, ymax), 255, -1)

# page_outline corners: TL, BL, BR, TR
page_outline = np.array([
    [xmin, ymin],
    [xmin, ymax],
    [xmax, ymax],
    [xmax, ymin],
], dtype=np.float32)

name = "IMG_1389"

# Detect contours using Mask
mask_text = Mask(name, small, pagemask, text=True)
contour_list = get_contours(name, small, mask_text.value)
print(f"Detected {len(contour_list)} contours (text mode)")

spans = assemble_spans(name, small, pagemask, contour_list)
if len(spans) < 3:
    mask_line = Mask(name, small, pagemask, text=False)
    contour_list2 = get_contours(name, small, mask_line.value)
    spans2 = assemble_spans(name, small, pagemask, contour_list2)
    if len(spans2) > len(spans):
        contour_list = contour_list2
        spans = spans2
print(f"Assembled {len(spans)} spans")

if not spans:
    print("No spans found!")
    sys.exit(1)

span_points = sample_spans(small.shape, spans)
print(f"Sampled span points: {len(span_points)} spans")

corners, ycoords, xcoords = keypoints_from_samples(
    name, small, pagemask, page_outline, span_points
)
rough_dims, span_counts, params = get_default_params(corners, ycoords, xcoords)
print(f"Initial params shape: {params.shape}")
print(f"Initial rvec: {params[0:3]}")
print(f"Initial tvec: {params[3:6]}")
print(f"span_counts: {span_counts}")

dstpoints = np.vstack((corners[0].reshape((1, 1, 2)),) + tuple(span_points))
print(f"dstpoints shape: {dstpoints.shape}")

keypoint_index = make_keypoint_index(span_counts)
objective = make_objective(dstpoints, keypoint_index, cfg.SHEAR_COST, slice(*cfg.RVEC_IDX))

print(f"Initial objective: {objective(params):.6f}")

# ------- Run with different maxcor values and collect iterations --------

def run_lbfgsb(params, maxcor, label):
    """Run L-BFGS-B and collect per-iteration data."""
    iterations = []
    iter_count = [0]

    def callback(xk):
        iter_count[0] += 1
        f = objective(xk)
        entry = {
            'iter': iter_count[0],
            'f': float(f),
            'rvec': xk[0:3].tolist(),
            'tvec': xk[3:6].tolist(),
            'cubic': xk[6:8].tolist(),
        }
        iterations.append(entry)
        if iter_count[0] <= 5 or iter_count[0] % 100 == 0:
            print(f"  [{label}] iter {iter_count[0]:4d}: f={f:.6f} rvec[0]={xk[0]:.6f}")

    result = minimize(
        objective,
        params.copy(),
        method='L-BFGS-B',
        options={'maxiter': 600000, 'maxcor': maxcor},
        callback=callback,
    )
    print(f"[{label}] Done: {result.nfev} evals, f={result.fun:.6f}, rvec[0]={result.x[0]:.6f}")
    return result, iterations

print("\n--- Running L-BFGS-B with maxcor=10 (scipy default) ---")
result10, iters10 = run_lbfgsb(params, maxcor=10, label='maxcor=10')

print("\n--- Running L-BFGS-B with maxcor=100 (Swift/page-dewarp config) ---")
result100, iters100 = run_lbfgsb(params, maxcor=100, label='maxcor=100')

# Save trace
trace_data = {
    'image': IMG_PATH,
    'initial_params': params.tolist(),
    'initial_f': float(objective(params)),
    'span_counts': span_counts,
    'param_count': int(params.shape[0]),
    'maxcor10': {
        'final_x': result10.x.tolist(),
        'final_f': float(result10.fun),
        'nfev': result10.nfev,
        'nit': result10.nit,
        'rvec': result10.x[0:3].tolist(),
        'tvec': result10.x[3:6].tolist(),
        'iterations': iters10,
    },
    'maxcor100': {
        'final_x': result100.x.tolist(),
        'final_f': float(result100.fun),
        'nfev': result100.nfev,
        'nit': result100.nit,
        'rvec': result100.x[0:3].tolist(),
        'tvec': result100.x[3:6].tolist(),
        'iterations': iters100,
    },
}

trace_file = DEBUG_DIR / 'python_trace.json'
with open(trace_file, 'w') as f:
    json.dump(trace_data, f, indent=2)
print(f"\nSaved trace to {trace_file}")

print("\n=== KEY FINDING SUMMARY ===")
print(f"Initial rvec:            {params[0:3]}")
print(f"maxcor=10  rvec[0]:      {result10.x[0]:.6f}  (scipy default)")
print(f"maxcor=100 rvec[0]:      {result100.x[0]:.6f}  (Swift config)")
print(f"Expected Python result:  rvec[0]≈0.053")
print(f"Expected Swift result:   rvec[0]≈0.184")
print(f"\nNote: Python scipy L-BFGS-B does NOT pass gradient (uses finite differences)")
print(f"      Swift L-BFGS-B uses ANALYTICAL gradient (objectiveAndGradient)")
