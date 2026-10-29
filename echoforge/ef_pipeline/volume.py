"""
LV volume from a binary mask by the disk-summation (method of discs) approximation.

Adapted from `maskvolume.py` (P. Naidoo, 2022) and `drawhelper.py` of the original in-house
code; made dependency-light (no sympy, no star imports) and independent of the mask size.
Returns the volume in pixel^3: multiply by pixel_spacing_cm**3 to get mL.
"""
import math

import cv2
import numpy as np
from skimage.draw import line


def _dist(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def _mid(p1, p2):
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def _norm(v):
    d = math.sqrt(v[0] * v[0] + v[1] * v[1])
    return (v[0] / d, v[1] / d)


def _point_at(p, v, d):
    return (p[0] + v[0] * d, p[1] + v[1] * d)


def _intersections(canvas, poly, pt1, pt2):
    """Pixels where the line pt1->pt2 crosses the contour `poly` (marked on a copy of `canvas`)."""
    img = canvas.copy()
    pts = np.array(poly, np.int32).reshape((-1, 1, 2))
    img = cv2.line(img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (255, 0, 0), thickness=2)
    img = cv2.fillPoly(img, [pts], (0, 255, 0))
    img = cv2.polylines(img, [pts], True, (255, 255, 0), thickness=1)

    discrete = list(zip(*line(int(pt1[0]), int(pt1[1]), int(pt2[0]), int(pt2[1]))))
    hits, prev = [], (0, 0, 0)
    for i, (x, y) in enumerate(discrete):
        if abs(y) >= img.shape[0] or abs(x) >= img.shape[1] or x <= 0 or y <= 0:
            continue
        pix = img[int(y), int(x)]
        if np.array_equal(pix, (255, 255, 0)):
            hits.append((x, y))
        elif i > 0:
            if np.array_equal(prev, (255, 0, 0)) and np.array_equal(pix, (0, 255, 0)):
                hits.append((x, y))
            elif np.array_equal(prev, (0, 255, 0)) and np.array_equal(pix, (255, 0, 0)):
                hits.append((x, y))
        prev = pix
    return hits


def mask_volume(mask, k=20):
    """Volume (pixel^3) of a binary LV mask (2-D array of 0/1), using `k` discs along the long axis."""
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask_rgb = np.repeat((mask > 0).astype(np.uint8)[..., None] * 255, 3, axis=2)

    contours, _ = cv2.findContours(mask_rgb[..., 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    poly = max(contours, key=cv2.contourArea)
    if cv2.contourArea(poly) <= 0:
        return 0.0

    x, y, w, h = cv2.boundingRect(poly)
    min_col, max_col, max_row = x - w, x + 2 * w, y + 2 * h

    pts = poly[:, 0, :].astype(np.float64)
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = (diff ** 2).sum(-1)
    np.fill_diagonal(d2, -1)
    i1, i2 = np.unravel_index(np.argmax(d2), d2.shape)
    dr = math.sqrt(d2[i1, i2])
    left = pts[np.argmin(np.hypot(pts[:, 0] - min_col, pts[:, 1] - max_row))]
    right = pts[np.argmin(np.hypot(pts[:, 0] - max_col, pts[:, 1] - max_row))]

    c1, c2 = poly[i1][0], poly[i2][0]
    min_p, max_p = (c1, c2) if c1[1] > c2[1] else (c2, c1)  # max_p = apex (smaller y)

    mid_temp = _mid(left, right)
    v = _norm((max_p[0] - mid_temp[0], max_p[1] - mid_temp[1]))
    hits = _intersections(mask_rgb, poly, _point_at(max_p, v, dr), _point_at(max_p, v, -dr * 1.10))
    if len(hits) >= 2:
        true_mid = hits[0] if _dist(mid_temp, hits[0]) < _dist(mid_temp, hits[-1]) else hits[-1]
    else:
        true_mid = mid_temp

    start_pt, end_pt = max_p, true_mid
    v = _norm((end_pt[0] - start_pt[0], end_pt[1] - start_pt[1]))
    step = _dist(start_pt, end_pt) / k

    blank = np.zeros_like(mask_rgb)
    vol, start = 0.0, (float(start_pt[0]), float(start_pt[1]))
    while True:
        end = _point_at(start, v, step)
        vh = (end[0] - start[0], end[1] - start[1])
        start = end
        if _dist(start_pt, end) >= dr * 1.1:
            break
        perp = _norm((vh[1], -vh[0]))
        hits = _intersections(blank, poly, _point_at(end, perp, -dr), _point_at(end, perp, dr))
        if len(hits) < 2:
            break
        width = _dist(hits[0], hits[-1])
        vol += math.pi * (width / 2.0) ** 2 * step
    return vol
