"""
EchoForge EF pipeline: A4C echo cine (DICOM or video) -> multi-beat ED/ES detection (EchoPDNet)
-> LV segmentation (EchoLVSNet) -> disk-summation volumes -> ejection fraction per beat.

    from echoforge.ef_pipeline import compute_ef
    result = compute_ef("study.dcm")
    print(result["EF_mean"])
"""
import json
import os
import shutil
import subprocess
import traceback

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
PREVIEW_MAX_SIDE = 480
FALLBACK_FPS = 30.0


def _report_progress(progress, stage, fraction):
    if progress is not None:
        progress(stage, fraction)


# ---------------------------------------------------------------- input
def _positive_number(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number <= 0 or number != number:
        return None
    return number


def _dicom_fps(ds):
    """Frame rate from DICOM timing tags, or None when the file does not carry one."""
    for name in ("RecommendedDisplayFrameRate", "CineRate"):
        fps = _positive_number(getattr(ds, name, None))
        if fps:
            return round(fps, 2)
    frame_time_ms = _positive_number(getattr(ds, "FrameTime", None))
    if frame_time_ms:
        return round(1000.0 / frame_time_ms, 2)
    return None


def _is_stored_ybr(frame):
    """True when a 3-channel frame is still YCbCr rather than RGB.

    pydicom converts YBR_FULL and YBR_FULL_422 to RGB and leaves the
    Photometric Interpretation tag unchanged. Cb and Cr of real YCbCr sit near
    128, while the brightness channel of an echo does not.
    """
    means = frame.reshape(-1, frame.shape[-1])[:, :3].mean(axis=0)
    chroma_centered = abs(float(means[1]) - 128) < 20 and abs(float(means[2]) - 128) < 20
    return chroma_centered and abs(float(means[0]) - float(means[1])) > 30


def _gray_frame(frame, photometric):
    """One uint8 grayscale frame. A YBR tag on RGB pixels is left as RGB."""
    frame = np.asarray(frame)
    if frame.ndim == 3 and frame.shape[-1] == 1:
        frame = frame[..., 0]
    if frame.ndim == 3:
        frame = np.ascontiguousarray(frame[..., :3])
        if str(photometric).startswith("YBR") and _is_stored_ybr(frame):
            # DICOM order is Y, Cb, Cr. OpenCV's YCrCb is Y, Cr, Cb.
            frame = cv2.cvtColor(np.ascontiguousarray(frame[..., [0, 2, 1]]), cv2.COLOR_YCrCb2RGB)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        frame = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2GRAY)
    elif frame.dtype != np.uint8:
        peak = float(frame.max()) if frame.size else 0.0
        frame = np.zeros(frame.shape[:2], np.uint8) if peak <= 0 else (
            frame.astype(np.float32) / peak * 255.0
        ).astype(np.uint8)
    return np.ascontiguousarray(frame)


def read_dicom(path):
    """Return (grayscale uint8 frames, pixel spacing in cm or None, fps or None) from a DICOM cine."""
    import pydicom
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array
    if int(getattr(ds, "NumberOfFrames", 1)) == 1:
        arr = arr[None]
    photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2"))
    frames = [_gray_frame(f, photometric) for f in arr]

    spacing = None
    if "PixelSpacing" in ds:
        spacing = float(ds.PixelSpacing[0]) / 10.0  # mm -> cm
    else:
        try:  # PhysicalDeltaX is already in cm
            spacing = float(ds.SequenceOfUltrasoundRegions[0].PhysicalDeltaX)
        except Exception:
            pass
    return frames, spacing, _dicom_fps(ds)


def read_video(path, pixel_spacing_mm=None):
    """Return (frames, spacing_cm or None, fps or None) from a video file. A video carries no spacing:
    it is taken from a same-named .dcm next to it, else from `pixel_spacing_mm`, else None."""
    cap = cv2.VideoCapture(path)
    frames = []
    fps = _positive_number(cap.get(cv2.CAP_PROP_FPS))
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
    return frames, spacing, round(fps, 2) if fps else None


def read_input(path, pixel_spacing_mm=None):
    if path.lower().endswith(".dcm"):
        return read_dicom(path)
    return read_video(path, pixel_spacing_mm)


# ------------------------------------------------------- phase detection
def predict_phase_curve(frames, pd_model, batch_size=16, *, progress=None):
    """Sliding-window (stride 1) EchoPDNet inference, averaged over all windows covering a frame."""
    _report_progress(progress, "phase", 0.0)
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
        _report_progress(progress, "phase", (i + len(bs)) / len(starts))
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


def _preview_frame(frame):
    """Shrink a grayscale frame so a JPEG fallback stays small."""
    height, width = frame.shape[:2]
    scale = min(1.0, PREVIEW_MAX_SIDE / max(height, width))
    if scale < 1:
        frame = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))),
                           interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(frame)


def _as_gray(frame):
    """One uint8 grayscale frame, even if a DICOM frame arrived as RGB or 16-bit."""
    frame = np.asarray(frame)
    if frame.ndim == 3:
        frame = frame[..., 0] if frame.shape[2] == 1 else cv2.cvtColor(frame[..., :3], cv2.COLOR_BGR2GRAY)
    if frame.dtype != np.uint8:
        peak = float(np.max(frame)) if frame.size else 0.0
        frame = np.zeros(frame.shape[:2], np.uint8) if peak <= 0 else (frame.astype(np.float32) / peak * 255.0).astype(np.uint8)
    return np.ascontiguousarray(frame)


def _write_mp4(frames, path, fps):
    """Encode grayscale frames to H.264. Returns False when ffmpeg cannot do it."""
    frames = [_as_gray(frame) for frame in frames]
    height, width = frames[0].shape[:2]
    even_h, even_w = height + (height % 2), width + (width % 2)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "gray",
        "-s", f"{even_w}x{even_h}", "-r", f"{fps}",
        "-i", "pipe:0",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for frame in frames:
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            padded = frame
            if frame.shape[0] != even_h or frame.shape[1] != even_w:
                padded = np.zeros((even_h, even_w), np.uint8)
                padded[:frame.shape[0], :frame.shape[1]] = frame
            proc.stdin.write(np.ascontiguousarray(padded).tobytes())
        proc.stdin.close()
        err = proc.stderr.read()
        code = proc.wait(timeout=180)
    except Exception:
        proc.kill()
        proc.wait(timeout=10)
        raise
    if code != 0:
        raise RuntimeError(err.decode("utf-8", "replace")[-500:])
    return os.path.isfile(path) and os.path.getsize(path) > 0


def _write_preview(frames, output_dir, fps):
    """Browser-playable cine: H.264 MP4 when ffmpeg is available, otherwise small JPEGs.

    DICOM and many echo codecs will not play in a browser, so this is encoded from the
    frames already decoded for analysis. Returns a dict the API turns into URLs, or None.
    """
    if not frames:
        return None
    playback = float(fps) if fps and fps > 0 else FALLBACK_FPS
    mp4_path = os.path.join(output_dir, "preview.mp4")
    if shutil.which("ffmpeg"):
        try:
            if _write_mp4(frames, mp4_path, playback):
                return {"kind": "mp4", "file": "preview.mp4", "fps": round(playback, 2)}
        except Exception:
            traceback.print_exc()
            try:
                os.remove(mp4_path)
            except OSError:
                pass
    files = []
    try:
        for i, frame in enumerate(frames):
            name = f"preview_{i:04d}.jpg"
            cv2.imwrite(os.path.join(output_dir, name), _preview_frame(frame),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            files.append(name)
    except Exception:
        traceback.print_exc()
        return None
    if not files:
        return None
    return {"kind": "jpeg", "files": files, "fps": round(playback, 2)}


# ------------------------------------------------------------------ main
def compute_ef(input_path, pixel_spacing_mm=None, output_dir=None, pd_model=None, seg_model=None, *, progress=None):
    """Compute LVEF for one A4C cine.

    Args:
        input_path: DICOM (.dcm) or video (.avi, .mp4, ...).
        pixel_spacing_mm: only used for a video with no same-named .dcm; otherwise volumes are px^3.
        output_dir: if given, it is emptied and filled with ED/ES frames, masks, phase_curve.txt,
            result.json, and a browser preview (preview.mp4, or preview_XXXX.jpg if ffmpeg is missing).
        pd_model, seg_model: optionally pass already-loaded models to reuse them across calls.
        progress: optional callback(stage, fraction) with progress within each EF stage.

    Returns:
        dict with `beats` (ED/ES frame, EDV, ESV, EF per beat), `EF_mean`, `EF_median`, `unit`,
        `pixel_spacing_cm` (None when volumes are px^3),
        `fps`, `geometry` (long-axis points per ED/ES frame), and `preview` when `output_dir` is set.
    """
    from echoforge import load_model

    frames, spacing_cm, fps = read_input(input_path, pixel_spacing_mm)
    pd_model = pd_model or load_model("EchoPDNet", pretrained=True)
    curve = predict_phase_curve(frames, pd_model, progress=progress)
    ed_list, es_list = detect_beats(curve)
    pairs = pair_beats(ed_list, es_list)
    if not pairs:
        raise RuntimeError("No complete ED->ES beat found in this video")

    seg_model = seg_model or load_model("EchoLVSNet", pretrained=True)
    scale = spacing_cm ** 3 if spacing_cm else 1.0
    unit = "mL" if spacing_cm else "px^3"

    if output_dir:
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    cache = {}
    geometry = {}
    _report_progress(progress, "segmentation", 0.0)
    mask_count = len({index for pair in pairs for index in pair})

    def volume(idx):
        if idx not in cache:
            mask = segment_frame(frames[idx], seg_model)
            vol_px, geom = mask_volume(mask, K_DISCS)
            cache[idx] = vol_px * scale
            geometry[str(idx)] = geom
            _report_progress(progress, "segmentation", len(cache) / mask_count)
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

    _report_progress(progress, "volumes", 0.0)
    efs = [b["EF_percent"] for b in beats if b["valid"]]
    result = {"input": str(input_path), "num_frames": len(frames), "fps": fps, "unit": unit,
              "pixel_spacing_cm": spacing_cm,
              "ED_frames": ed_list, "ES_frames": es_list, "beats": beats,
              "geometry": geometry,
              "EF_mean": _r2(np.mean(efs)) if efs else None,
              "EF_median": _r2(np.median(efs)) if efs else None}
    if output_dir:
        preview = _write_preview(frames, output_dir, fps)
        if preview:
            result["preview"] = preview
        np.savetxt(os.path.join(output_dir, "phase_curve.txt"), curve, fmt="%.2f")
        with open(os.path.join(output_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=2)
    _report_progress(progress, "volumes", 1.0)
    return result
