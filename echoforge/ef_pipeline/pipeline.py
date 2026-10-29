"""
EchoForge EF pipeline: A4C echo cine (DICOM or video) -> multi-beat ED/ES detection (EchoPDNet)
-> LV segmentation (EchoLVSNet) -> disk-summation volumes -> ejection fraction per beat.

    from echoforge.ef_pipeline import compute_ef
    result = compute_ef("study.dcm")
    print(result["EF_mean"])
"""
import json
import os

import cv2
import numpy as np
from scipy.signal import find_peaks

from .volume import mask_volume

SEQ_LEN = 30          # EchoPDNet window length (frames)
PD_SIZE = 112         # EchoPDNet input size
SEG_SIZE = 512        # EchoLVSNet input size
SEG_THRESHOLD = 0.5
K_DISCS = 20
MIN_GAP = 0           # frames to skip after an ED before accepting its ES (0 = no filtering)


# ---------------------------------------------------------------- input
def read_dicom(path):
    """Return (list of grayscale uint8 frames, pixel spacing in cm or None) from a DICOM cine."""
    import pydicom
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array
    if int(getattr(ds, "NumberOfFrames", 1)) == 1:
        arr = arr[None]
    photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2"))
    frames = []
    for f in arr:
        if f.ndim == 3:
            if photometric.startswith("YBR"):
                f = cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
            f = cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        elif f.dtype != np.uint8:
            f = (f / max(float(f.max()), 1.0) * 255).astype(np.uint8)
        frames.append(f)

    spacing = None
    if "PixelSpacing" in ds:
        spacing = float(ds.PixelSpacing[0]) / 10.0  # mm -> cm
    else:
        try:  # PhysicalDeltaX is already in cm
            spacing = float(ds.SequenceOfUltrasoundRegions[0].PhysicalDeltaX)
        except Exception:
            pass
    return frames, spacing


def read_video(path, pixel_spacing_mm=None):
    """Return (frames, spacing_cm or None) from a video file. A video carries no spacing: it is taken
    from a same-named .dcm next to it, else from `pixel_spacing_mm`, else None."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {path}")
    spacing = None
    stem = os.path.splitext(path)[0]
    for ext in (".dcm", ".DCM"):
        if os.path.isfile(stem + ext):
            spacing = read_dicom(stem + ext)[1]
            break
    if spacing is None and pixel_spacing_mm:
        spacing = pixel_spacing_mm / 10.0
    return frames, spacing


def read_input(path, pixel_spacing_mm=None):
    if path.lower().endswith(".dcm"):
        return read_dicom(path)
    return read_video(path, pixel_spacing_mm)


# ------------------------------------------------------- phase detection
def predict_phase_curve(frames, pd_model, batch_size=16):
    """Sliding-window (stride 1) EchoPDNet inference, averaged over all windows covering a frame."""
    n = len(frames)
    small = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR), (PD_SIZE, PD_SIZE)) for f in frames]
    x = np.asarray(small, dtype=np.float32) / 255.0
    if n < SEQ_LEN:
        x = np.concatenate([x, np.zeros((SEQ_LEN - n, PD_SIZE, PD_SIZE, 3), np.float32)])
    total = len(x)
    starts = list(range(0, total - SEQ_LEN + 1))
    acc, cnt = np.zeros(total), np.zeros(total)
    for i in range(0, len(starts), batch_size):
        bs = starts[i:i + batch_size]
        preds = pd_model.predict(np.stack([x[s:s + SEQ_LEN] for s in bs]), verbose=0).reshape(len(bs), SEQ_LEN)
        for s, p in zip(bs, preds):
            acc[s:s + SEQ_LEN] += p
            cnt[s:s + SEQ_LEN] += 1
    return (acc / cnt)[:n]


def detect_beats(curve, distance=10, prominence=0.05):
    """ED = maxima, ES = minima of the min-max normalised phase curve (works for multiple beats)."""
    p = (curve - curve.min()) / max(curve.max() - curve.min(), 1e-8)
    ed, _ = find_peaks(p, distance=distance, prominence=prominence)
    es, _ = find_peaks(-p, distance=distance, prominence=prominence)
    return ed.tolist(), es.tolist()


def pair_beats(ed_list, es_list, min_gap=MIN_GAP):
    """Pair each ED with the first ES after it and before the next ED."""
    pairs = []
    for i, ed in enumerate(ed_list):
        nxt = ed_list[i + 1] if i + 1 < len(ed_list) else np.inf
        cands = [e for e in es_list if e - ed > min_gap and e < nxt]
        if cands:
            pairs.append((ed, cands[0]))
    return pairs


# ---------------------------------------------------------- segmentation
def segment_frame(frame, seg_model):
    """Pad to 512x512 keeping aspect ratio, run EchoLVSNet, and map the binary mask back to the frame size."""
    h, w = frame.shape
    s = SEG_SIZE / max(h, w)
    nh, nw = int(h * s), int(w * s)
    top, left = (SEG_SIZE - nh) // 2, (SEG_SIZE - nw) // 2
    padded = np.zeros((SEG_SIZE, SEG_SIZE), np.uint8)
    padded[top:top + nh, left:left + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    # EchoLVSNet rescales by 1/255 internally, so it gets raw 0-255 values
    pred = seg_model.predict(padded[None, ..., None].astype(np.float32), verbose=0)[0]
    if pred.ndim == 3:
        pred = pred[..., 0]
    small = (pred >= SEG_THRESHOLD).astype(np.uint8)[top:top + nh, left:left + nw]
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def _r2(v):
    return None if v is None else round(float(v), 2)


# ------------------------------------------------------------------ main
def compute_ef(input_path, pixel_spacing_mm=None, output_dir=None, pd_model=None, seg_model=None):
    """Compute LVEF for one A4C cine.

    Args:
        input_path: DICOM (.dcm) or video (.avi, .mp4, ...).
        pixel_spacing_mm: only used for a video with no same-named .dcm; otherwise volumes are px^3.
        output_dir: if given, it is emptied and filled with ED/ES frames, masks, phase_curve.txt, result.json.
        pd_model, seg_model: optionally pass already-loaded models to reuse them across calls.

    Returns:
        dict with `beats` (ED/ES frame, EDV, ESV, EF per beat), `EF_mean`, `EF_median`, `unit`, ...
    """
    from echoforge import load_model

    frames, spacing_cm = read_input(input_path, pixel_spacing_mm)
    pd_model = pd_model or load_model("EchoPDNet", pretrained=True)
    curve = predict_phase_curve(frames, pd_model)
    ed_list, es_list = detect_beats(curve)
    pairs = pair_beats(ed_list, es_list)
    if not pairs:
        raise RuntimeError("No complete ED->ES beat found in this video")

    seg_model = seg_model or load_model("EchoLVSNet", pretrained=True)
    scale = spacing_cm ** 3 if spacing_cm else 1.0
    unit = "mL" if spacing_cm else "px^3"

    if output_dir:
        import shutil
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    cache = {}

    def volume(idx):
        if idx not in cache:
            mask = segment_frame(frames[idx], seg_model)
            cache[idx] = mask_volume(mask, K_DISCS) * scale
            if output_dir:
                cv2.imwrite(os.path.join(output_dir, f"frame_{idx:04d}_mask.png"), mask * 255)
                cv2.imwrite(os.path.join(output_dir, f"frame_{idx:04d}.png"), frames[idx])
        return cache[idx]

    beats = []
    for ed, es in pairs:
        edv, esv = volume(ed), volume(es)
        ef = (edv - esv) / edv * 100.0 if edv > 0 else None
        beats.append({"ED_frame": ed, "ES_frame": es, f"EDV_{unit}": _r2(edv), f"ESV_{unit}": _r2(esv),
                      "EF_percent": _r2(ef), "valid": bool(ef is not None and edv > esv > 0)})

    efs = [b["EF_percent"] for b in beats if b["valid"]]
    result = {"input": str(input_path), "num_frames": len(frames), "unit": unit,
              "ED_frames": ed_list, "ES_frames": es_list, "beats": beats,
              "EF_mean": _r2(np.mean(efs)) if efs else None,
              "EF_median": _r2(np.median(efs)) if efs else None}
    if output_dir:
        np.savetxt(os.path.join(output_dir, "phase_curve.txt"), curve, fmt="%.2f")
        with open(os.path.join(output_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=2)
    return result
