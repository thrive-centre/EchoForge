"""
Phase Detection + MCLV EF pipeline.

A4C cine -> EchoPDNet phase curve -> ED/ES frames -> MCLV masks ->
disk-summation volumes -> EF for every detected beat.
"""
import json
import os

import cv2
import numpy as np

from .pipeline import (
    read_input,
    detect_beats,
    pair_beats,
    predict_phase_curve,
)
from .mclv_pipeline import segment_frames as segment_mclv_frames
from .volume import mask_volume


def compute_phase_mclv_ef(
    input_path,
    pixel_spacing_mm=None,
    output_dir=None,
    pd_model=None,
    seg_model=None,
):
    """Compute EF for all beats using EchoPDNet and MCLV."""
    from echoforge import load_model

    frames, spacing_cm, _fps = read_input(input_path, pixel_spacing_mm)
    pd_model = pd_model or load_model("EchoPDNet", pretrained=True)
    seg_model = seg_model or load_model("MCLV", pretrained=True)

    curve = predict_phase_curve(frames, pd_model)
    ed_list, es_list = detect_beats(curve)
    pairs = pair_beats(ed_list, es_list)
    if not pairs:
        raise RuntimeError("No complete ED-to-ES beat found in this video")

    cache = {}
    masks = {}

    def volume(index):
        if index not in cache:
            mask = segment_mclv_frames([frames[index]], seg_model)[0]
            masks[index] = mask
            cache[index] = (
                float(mask_volume(mask, 20)[0]) if mask.any() else 0.0
            )
        return cache[index]

    scale = spacing_cm ** 3 if spacing_cm else 1.0
    unit = "mL" if spacing_cm else "px^3"
    beats = []
    for ed, es in pairs:
        edv = volume(ed) * scale
        esv = volume(es) * scale
        ef = (edv - esv) / edv * 100.0 if edv > 0 else None
        valid = bool(ef is not None and edv > esv > 0)
        beats.append({
            "ED_frame": ed,
            "ES_frame": es,
            f"EDV_{unit}": round(float(edv), 2),
            f"ESV_{unit}": round(float(esv), 2),
            "EF_percent": round(float(ef), 2) if ef is not None else None,
            "valid": valid,
        })

    efs = [beat["EF_percent"] for beat in beats if beat["valid"]]
    ed_frames = [beat["ED_frame"] for beat in beats]
    es_frames = [beat["ES_frame"] for beat in beats]
    result = {
        "input": str(input_path),
        "phase_model": "EchoPDNet",
        "segmentation_model": "MCLV",
        "beat_detection": "phase-model",
        "volume_method": "disk-summation",
        "num_frames": len(frames),
        "unit": unit,
        "ED_frames": ed_list,
        "ES_frames": es_list,
        "beats": beats,
        "EF_mean": round(float(np.mean(efs)), 2) if efs else None,
        "EF_median": round(float(np.median(efs)), 2) if efs else None,
        "phase_curve": [round(float(value), 6) for value in curve],
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
            os.path.join(output_dir, "phase_curve.txt"),
            curve,
            fmt="%.2f",
        )
        with open(os.path.join(output_dir, "result.json"), "w") as file:
            json.dump(result, file, indent=2)
    return result
