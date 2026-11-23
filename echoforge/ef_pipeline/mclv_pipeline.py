"""
MCLV EF pipeline: A4C echo cine -> LV segmentation with MCLV ->
disk-summation volume trajectory -> ED/ES of the first beat -> ejection fraction.

    from echoforge.ef_pipeline import compute_mclv_ef
    result = compute_mclv_ef("study.dcm")
"""
import json
import os

import cv2
import numpy as np

from .pipeline import _report_progress, read_input
from .volume import mask_volume
from .volume_methods import select_beats


SEG_SIZE = 256
SEG_THRESHOLD = 0.5
K_DISCS = 20
OUTLIER_WIN = 5
OUTLIER_SIGMAS = 3.0
MIN_PROMINENCE = 0.3
BATCH_SIZE = 4
MIN_FRAMES_EARLY_STOP = 15
CONFIRM_FRAMES = 3
CONFIRM_RISE = 0.15


def _mask_volume(mask):
    try:
        return float(mask_volume(mask, K_DISCS)[0]) if mask.any() else 0.0
    except Exception:
        return 0.0


def select_first_beat(vols):
    """Compatibility wrapper returning the first complete beat."""
    pairs, clean, outlier = select_beats(vols)
    ed, es = pairs[0]
    return ed, es, clean, outlier


def segment_frames(frames, seg_model):
    """Segment grayscale frames with MCLV and map masks to the original size."""
    batch = []
    geometry = []

    for frame in frames:
        h, w = frame.shape
        scale = SEG_SIZE / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        top, left = (SEG_SIZE - nh) // 2, (SEG_SIZE - nw) // 2
        padded = np.zeros((SEG_SIZE, SEG_SIZE), np.uint8)
        padded[top:top + nh, left:left + nw] = cv2.resize(
            frame, (nw, nh), interpolation=cv2.INTER_AREA
        )
        batch.append(padded)
        geometry.append((h, w, nh, nw, top, left))

    predictions = np.asarray(seg_model(
        np.asarray(batch, np.float32)[..., None],
        training=False,
    ))
    masks = []
    for prediction, (h, w, nh, nw, top, left) in zip(predictions, geometry):
        if prediction.ndim == 3:
            prediction = prediction[..., 0]
        small = (prediction >= SEG_THRESHOLD).astype(np.uint8)
        small = small[top:top + nh, left:left + nw]
        masks.append(
            cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        )
    return masks


def segment_frame(frame, seg_model):
    return segment_frames([frame], seg_model)[0]


def volume_curve(frames, seg_model, early_stop=False, batch_size=BATCH_SIZE, *, progress=None):
    """Segment every processed frame and calculate its LV volume trajectory."""
    masks = []
    volumes = []
    _report_progress(progress, "segmentation", 0.0)
    for i in range(0, len(frames), batch_size):
        for mask in segment_frames(frames[i:i + batch_size], seg_model):
            masks.append(mask)
            volumes.append(_mask_volume(mask))
        _report_progress(progress, "segmentation", len(masks) / len(frames))
        if early_stop and len(volumes) >= len(frames):
            break
    return masks, np.asarray(volumes, dtype=float)


def _r2(value):
    return None if value is None else round(float(value), 2)


def compute_mclv_ef(
    input_path,
    pixel_spacing_mm=None,
    output_dir=None,
    seg_model=None,
    early_stop=True,
    *,
    progress=None,
):
    """Compute LVEF for all detected beats using MCLV and disk-summation volumes."""
    from echoforge import load_model

    frames, spacing_cm, _fps = read_input(input_path, pixel_spacing_mm)
    seg_model = seg_model or load_model("MCLV", pretrained=True)
    scale = spacing_cm ** 3 if spacing_cm else 1.0
    unit = "mL" if spacing_cm else "px^3"

    masks, raw_volumes = volume_curve(
        frames,
        seg_model,
        early_stop=early_stop,
        progress=progress,
    )
    _report_progress(progress, "volumes", 0.0)
    pairs, clean_volumes, outlier = select_beats(raw_volumes)
    curve = clean_volumes * scale
    beats = []
    for ed, es in pairs:
        edv = float(curve[ed])
        esv = float(curve[es])
        ef = (edv - esv) / edv * 100.0 if edv > 0 else None
        valid = bool(ef is not None and edv > esv > 0)
        beats.append({
            "ED_frame": ed,
            "ES_frame": es,
            f"EDV_{unit}": _r2(edv),
            f"ESV_{unit}": _r2(esv),
            "EF_percent": _r2(ef),
            "valid": valid,
        })
    efs = [beat["EF_percent"] for beat in beats if beat["valid"]]
    ed_frames = [beat["ED_frame"] for beat in beats]
    es_frames = [beat["ES_frame"] for beat in beats]
    result = {
        "input": str(input_path),
        "segmentation_model": "MCLV",
        "volume_method": "disk-summation",
        "num_frames": len(frames),
        "frames_analyzed": len(raw_volumes),
        "unit": unit,
        "ED_frames": ed_frames,
        "ES_frames": es_frames,
        "beats": beats,
        "EF_mean": _r2(np.mean(efs)) if efs else None,
        "EF_median": _r2(np.median(efs)) if efs else None,
        "volume_curve": [_r2(value) for value in curve],
        "outlier_frames": np.flatnonzero(outlier).tolist(),
    }

    if output_dir:
        import shutil
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        for index in sorted(set(ed_frames + es_frames)):
            cv2.imwrite(
                os.path.join(output_dir, f"frame_{index:04d}_mask.png"),
                masks[index] * 255,
            )
            cv2.imwrite(
                os.path.join(output_dir, f"frame_{index:04d}.png"),
                frames[index],
            )
        np.savetxt(
            os.path.join(output_dir, "volume_curve.txt"),
            curve,
            fmt="%.2f",
        )
        with open(os.path.join(output_dir, "result.json"), "w") as file:
            json.dump(result, file, indent=2)
    _report_progress(progress, "volumes", 1.0)
    return result
