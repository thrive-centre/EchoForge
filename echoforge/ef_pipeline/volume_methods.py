import numpy as np
from scipy.signal import find_peaks

MIN_PROMINENCE = 0.3
OUTLIER_WIN = 5
OUTLIER_SIGMAS = 3.0


def clean_curve(volumes, win=OUTLIER_WIN, n_sigmas=OUTLIER_SIGMAS):
    """Interpolate empty masks and single-frame spikes in a volume trajectory."""
    values = np.asarray(volumes, dtype=float)
    if values.size == 0:
        raise RuntimeError("No frames available for volume analysis")

    pad = win // 2
    padded = np.pad(values, pad, mode="edge")
    median = np.array([
        np.median(padded[index:index + win])
        for index in range(len(values))
    ])
    sigma = 1.4826 * np.median(np.abs(values - median))
    threshold = n_sigmas * max(sigma, 0.02 * max(float(np.median(values)), 1e-8))
    outlier = (np.abs(values - median) > threshold) | (values <= 0)
    good = ~outlier
    if good.sum() < 2:
        raise RuntimeError("Segmentation failed on (almost) every frame")
    indices = np.arange(len(values))
    return np.interp(indices, indices[good], values[good]), outlier


def select_beats(volumes):
    """Return all ED/ES beat pairs detected from a cleaned volume trajectory."""
    clean, outlier = clean_curve(volumes)
    if len(clean) < 3:
        raise RuntimeError("Video is too short to detect a cardiac beat")

    smooth = np.convolve(
        np.pad(clean, 1, mode="edge"),
        np.ones(3) / 3,
        mode="valid",
    )
    lo, hi = np.percentile(smooth, [5, 95])
    volume_range = hi - lo
    if volume_range <= 0:
        raise RuntimeError("Flat volume curve: no cardiac motion detected")

    floor = smooth.min() - volume_range
    peaks, _ = find_peaks(
        np.r_[floor, smooth, floor],
        prominence=MIN_PROMINENCE * volume_range,
    )
    ed_frames = (peaks - 1).tolist()
    ed_frames = [
        ed for ed in ed_frames
        if 0 <= ed < len(smooth)
    ]

    pairs = []
    for i, ed in enumerate(ed_frames):
        next_ed = ed_frames[i + 1] if i + 1 < len(ed_frames) else len(smooth) - 1
        if next_ed <= ed:
            continue

        es = ed + int(np.argmin(smooth[ed:next_ed + 1]))
        if es == ed:
            continue

        ed = max(ed - 1, 0) + int(np.argmax(
            clean[max(ed - 1, 0):ed + 2]
        ))
        es = max(es - 1, 0) + int(np.argmin(
            clean[max(es - 1, 0):es + 2]
        ))
        pairs.append((ed, es))

    if not pairs:
        raise RuntimeError("No complete ED-to-ES beat found in the volume curve")
    return pairs, clean, outlier
