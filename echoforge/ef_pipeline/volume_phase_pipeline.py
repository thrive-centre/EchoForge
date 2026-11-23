"""
Volume-picked EF with a phase-model cross-check.

A4C cine -> MCLV mask on every frame -> disk-summation volume curve ->
ED = largest and ES = smallest LV volume in each cycle -> EF from the masks at those frames.
EchoPDNet runs alongside only to check the timing: a beat whose ED or ES is more than a
couple of frames from the phase model's is flagged for review, never moved or dropped.

    from echoforge.ef_pipeline import compute_volume_phase_ef
    result = compute_volume_phase_ef("study.dcm")
"""
import json
import os
import shutil

import cv2
import numpy as np

from .mclv_pipeline import segment_frames as segment_mclv_frames
from .pipeline import _r2, _report_progress, _write_preview, detect_beats, predict_phase_curve, read_input
from .volume import mask_volume
from .volume_methods import select_beats

K_DISCS = 20
BATCH_SIZE = 16
# The last ES must have this many frames after it, or the clip ended mid-systole.
MIN_TAIL_FRAMES = 3
# How far an outlier ED/ES frame may move to the best clean frame next to it.
OUTLIER_SEARCH = 2
# Allowed gap between volume-picked and phase-model frames, in seconds (and at least 2 frames).
AGREE_SECONDS = 0.067
AGREE_MIN_FRAMES = 2


def _volume_and_geometry(mask):
    if not mask.any():
        return 0.0, None
    try:
        vol, geom = mask_volume(mask, K_DISCS)
        return float(vol), geom
    except Exception:
        return 0.0, None


def _clean_frame(index, raw, outlier, pick):
    """`index` if its mask is usable, else the clean neighbour with the largest (pick=max)
    or smallest (pick=min) raw volume; None when no clean frame is close by."""
    if not outlier[index]:
        return index
    lo, hi = max(0, index - OUTLIER_SEARCH), min(len(raw), index + OUTLIER_SEARCH + 1)
    near = [i for i in range(lo, hi) if not outlier[i]]
    return pick(near, key=lambda i: raw[i]) if near else None


def _nearest(frame, candidates):
    if not candidates:
        return None
    return min(candidates, key=lambda c: abs(c - frame))


def compute_volume_phase_ef(
    input_path,
    pixel_spacing_mm=None,
    output_dir=None,
    pd_model=None,
    seg_model=None,
    *,
    progress=None,
):
    """Compute LVEF for every beat, picking ED/ES from the MCLV volume curve.

    Returns the usual result dict (`beats`, `EF_mean`, `EF_median`, `unit`, `fps`,
    `pixel_spacing_cm`, `geometry`, `preview`) plus `volume_curve` (cleaned, in `unit`),
    `outlier_frames`, and per beat the phase model's nearest `PD_ED_frame`/`PD_ES_frame`
    and `phase_agrees`.
    """
    from echoforge import load_model

    frames, spacing_cm, fps = read_input(input_path, pixel_spacing_mm)
    seg_model = seg_model or load_model("MCLV", pretrained=True)
    pd_model = pd_model or load_model("EchoPDNet", pretrained=True)
    scale = spacing_cm ** 3 if spacing_cm else 1.0
    unit = "mL" if spacing_cm else "px^3"

    masks, raw, geoms = [], [], []
    _report_progress(progress, "segmentation", 0.0)
    for i in range(0, len(frames), BATCH_SIZE):
        for mask in segment_mclv_frames(frames[i:i + BATCH_SIZE], seg_model):
            vol, geom = _volume_and_geometry(mask)
            masks.append(mask)
            raw.append(vol)
            geoms.append(geom)
        _report_progress(progress, "segmentation", len(masks) / len(frames))
    raw = np.asarray(raw, dtype=float)
    pairs, clean, outlier = select_beats(raw)

    phase = predict_phase_curve(frames, pd_model, progress=progress)
    pd_ed, pd_es = detect_beats(phase)
    tolerance = max(AGREE_MIN_FRAMES, int(round(AGREE_SECONDS * fps))) if fps else AGREE_MIN_FRAMES

    _report_progress(progress, "volumes", 0.0)
    beats = []
    for n, (ed, es) in enumerate(pairs):
        if n == len(pairs) - 1 and es >= len(frames) - MIN_TAIL_FRAMES:
            continue  # cut off by the end of the clip: the volume never rose again
        ed = _clean_frame(ed, raw, outlier, max)
        es = _clean_frame(es, raw, outlier, min)
        if ed is None or es is None or es <= ed:
            continue
        edv, esv = raw[ed] * scale, raw[es] * scale
        ef = (edv - esv) / edv * 100.0 if edv > 0 else None
        near_ed, near_es = _nearest(ed, pd_ed), _nearest(es, pd_es)
        agrees = (near_ed is not None and near_es is not None
                  and abs(near_ed - ed) <= tolerance and abs(near_es - es) <= tolerance)
        beats.append({
            "ED_frame": int(ed),
            "ES_frame": int(es),
            f"EDV_{unit}": _r2(edv),
            f"ESV_{unit}": _r2(esv),
            "EF_percent": _r2(ef),
            "valid": bool(ef is not None and edv > esv > 0),
            "PD_ED_frame": near_ed,
            "PD_ES_frame": near_es,
            "phase_agrees": bool(agrees),
        })
    if not beats:
        raise RuntimeError("No complete ED->ES beat found in the volume curve")

    key_frames = sorted({b["ED_frame"] for b in beats} | {b["ES_frame"] for b in beats})
    efs = [b["EF_percent"] for b in beats if b["valid"]]
    result = {
        "input": str(input_path),
        "segmentation_model": "MCLV",
        "beat_detection": "volume-curve",
        "phase_check_model": "EchoPDNet",
        "phase_tolerance_frames": tolerance,
        "volume_method": "disk-summation",
        "num_frames": len(frames),
        "fps": fps,
        "unit": unit,
        "pixel_spacing_cm": spacing_cm,
        "ED_frames": [b["ED_frame"] for b in beats],
        "ES_frames": [b["ES_frame"] for b in beats],
        "beats": beats,
        "geometry": {str(i): geoms[i] for i in key_frames if geoms[i]},
        "volume_curve": [_r2(v) for v in clean * scale],
        "outlier_frames": np.flatnonzero(outlier).tolist(),
        "EF_mean": _r2(np.mean(efs)) if efs else None,
        "EF_median": _r2(np.median(efs)) if efs else None,
    }

    if output_dir:
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        for i in key_frames:
            cv2.imwrite(os.path.join(output_dir, f"frame_{i:04d}_mask.png"), masks[i] * 255)
            cv2.imwrite(os.path.join(output_dir, f"frame_{i:04d}.png"), frames[i])
        preview = _write_preview(frames, output_dir, fps)
        if preview:
            result["preview"] = preview
        np.savetxt(os.path.join(output_dir, "phase_curve.txt"), phase, fmt="%.4f")
        np.savetxt(os.path.join(output_dir, "volume_curve.txt"), clean * scale, fmt="%.2f")
        with open(os.path.join(output_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=2)
    _report_progress(progress, "volumes", 1.0)
    return result
