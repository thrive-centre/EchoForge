"""
EchoLVSNet volume-only EF pipeline.

It derives ED/ES frames from the LV-volume trajectory; no EchoPDNet phase model is used.
"""
import cv2
import json
import os
import numpy as np

from .pipeline import read_input, segment_frame
from .volume import mask_volume
from .volume_methods import select_beats


def compute_echolvsnet_volume_ef(
    input_path,
    pixel_spacing_mm=None,
    output_dir=None,
    seg_model=None,
):
    """Compute all beats using EchoLVSNet masks and disk-summation volumes."""
    from echoforge import load_model

    frames, spacing_cm = read_input(input_path, pixel_spacing_mm)
    seg_model = seg_model or load_model("EchoLVSNet", pretrained=True)

    masks = []
    raw_volumes = []
    for i in range(0, len(frames), 4):
        batch_masks = [segment_frame(frame, seg_model) for frame in frames[i:i + 4]]
        masks.extend(batch_masks)
        raw_volumes.extend(
            float(mask_volume(mask, 20)) if mask.any() else 0.0
            for mask in batch_masks
        )

    pairs, clean, outlier = select_beats(np.asarray(raw_volumes, dtype=float))
    scale = spacing_cm ** 3 if spacing_cm else 1.0
    unit = "mL" if spacing_cm else "px^3"
    curve = clean * scale

    beats = []
    for ed, es in pairs:
        edv = float(curve[ed])
        esv = float(curve[es])
        ef = (edv - esv) / edv * 100.0 if edv > 0 else None
        valid = bool(ef is not None and edv > esv > 0)
        beats.append({
            "ED_frame": ed,
            "ES_frame": es,
            f"EDV_{unit}": round(edv, 2),
            f"ESV_{unit}": round(esv, 2),
            "EF_percent": round(float(ef), 2) if ef is not None else None,
            "valid": valid,
        })

    efs = [beat["EF_percent"] for beat in beats if beat["valid"]]
    ed_frames = [beat["ED_frame"] for beat in beats]
    es_frames = [beat["ES_frame"] for beat in beats]
    result = {
        "input": str(input_path),
        "segmentation_model": "EchoLVSNet",
        "beat_detection": "volume-trajectory",
        "volume_method": "disk-summation",
        "num_frames": len(frames),
        "frames_analyzed": len(raw_volumes),
        "unit": unit,
        "ED_frames": ed_frames,
        "ES_frames": es_frames,
        "beats": beats,
        "EF_mean": round(float(np.mean(efs)), 2) if efs else None,
        "EF_median": round(float(np.median(efs)), 2) if efs else None,
        "volume_curve": [round(float(value), 2) for value in curve],
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
    return result
