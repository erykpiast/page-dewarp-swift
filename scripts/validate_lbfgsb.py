#!/usr/bin/env python3
"""
Validate Python L-BFGS-B on all test images, recording rvec, pageDims, and loss.
Results saved to ~/Desktop/lbfgsb-debug/python_lbfgsb_results.json
"""

import json
import math
import os
import sys
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
from page_dewarp.projection import project_xy

DEBUG_DIR = Path.home() / 'Desktop' / 'lbfgsb-debug'
DEBUG_DIR.mkdir(exist_ok=True)

TEST_IMAGES = [
    Path.home() / 'Desktop' / 'IMG_1369.jpeg',
    Path.home() / 'Desktop' / 'IMG_1389.jpeg',
    Path.home() / 'Desktop' / 'IMG_1413.jpeg',
    Path.home() / 'Desktop' / 'IMG_1799.jpeg',
    Path.home() / 'Desktop' / 'IMG_1868.jpeg',
]

def run_lbfgsb(img_path):
    """Run L-BFGS-B pipeline on a single image, return metrics dict."""
    img_path = str(img_path)
    config = Config()

    cv2_img = cv2.imread(img_path)
    if cv2_img is None:
        return {"error": f"Could not load {img_path}"}

    # Resize to screen
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

    small_shape = small.shape[:2]

    # Page mask
    xmin = config.PAGE_MARGIN_X
    ymin = config.PAGE_MARGIN_Y
    xmax = small.shape[1] - xmin
    ymax = small.shape[0] - ymin
    pagemask = np.zeros(small_shape, dtype=np.uint8)
    cv2.rectangle(pagemask, (xmin, ymin), (xmax, ymax), 255, -1)
    page_outline = np.array([
        [xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]
    ], dtype=np.float32)

    # Contours (text mode)
    name = os.path.basename(img_path)
    mask = Mask(name, small, pagemask, text=True)
    contours = get_contours(name, small, mask.value)
    spans = assemble_spans(name, small, pagemask, contours)

    if len(spans) < 3:
        mask2 = Mask(name, small, pagemask, text=False)
        contours2 = get_contours(name, small, mask2.value)
        spans2 = assemble_spans(name, small, pagemask, contours2)
        if len(spans2) > len(spans):
            spans = spans2

    if not spans:
        return {"error": "no spans detected"}

    span_points = sample_spans(small_shape, spans)
    corners, ycoords, xcoords = keypoints_from_samples(name, small, pagemask, page_outline, span_points)

    rough_dims, span_counts, params0 = get_default_params(corners, ycoords, xcoords)

    # Build dstpoints
    keypoint_index = make_keypoint_index(span_counts)
    dstpoints = np.vstack((corners[0].reshape((1, 1, 2)),) + tuple(span_points))

    rvec_slice = slice(*cfg.RVEC_IDX)
    objective = make_objective(dstpoints, keypoint_index, cfg.SHEAR_COST, rvec_slice)

    # Run L-BFGS-B (no jacobian = finite differences, matching Swift fix)
    result = minimize(objective, params0, method='L-BFGS-B', options={'maxiter': 600000, 'maxcor': 10})

    params = result.x
    loss = float(result.fun)
    rvec = params[rvec_slice].tolist()

    # Get page dims
    dstBR = corners[2]
    def dim_objective(dims):
        proj = project_xy(np.array([dims]), params)
        # proj has shape (1, 1, 2) — project_xy returns (N, 1, 2)
        px, py = proj[0][0][0], proj[0][0][1]
        dx = dstBR[0][0] - px  # dstBR has shape (1,2), so [0][0] = x, [0][1] = y
        dy = dstBR[0][1] - py
        return dx*dx + dy*dy

    from scipy.optimize import minimize as scipy_min
    dim_res = scipy_min(dim_objective, rough_dims, method='Powell')
    page_dims = dim_res.x.tolist()

    return {
        "rvec": rvec,
        "rvec0": float(rvec[0]),
        "page_dims": page_dims,
        "loss": loss,
        "nfev": int(result.nfev),
        "success": bool(result.success),
        "message": str(result.message)
    }


results = {}
for img_path in TEST_IMAGES:
    name = img_path.name
    if not img_path.exists():
        print(f"SKIP {name}: not found")
        results[name] = {"error": "not found"}
        continue
    print(f"Processing {name}...", flush=True)
    try:
        r = run_lbfgsb(img_path)
        results[name] = r
        if "error" in r:
            print(f"  ERROR: {r['error']}")
        else:
            print(f"  rvec[0]={r['rvec0']:.4f}, pageDims={r['page_dims'][0]:.2f}x{r['page_dims'][1]:.2f}, loss={r['loss']:.6f}")
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results[name] = {"error": str(e)}

out_path = DEBUG_DIR / 'python_lbfgsb_results.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
