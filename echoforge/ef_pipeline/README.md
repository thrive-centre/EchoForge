# EF Pipeline Module

> **EchoForge EF Pipeline** computes the left-ventricular ejection fraction (LVEF) from a single apical four-chamber (A4C) echo cine. It chains two existing EchoForge models, **EchoPDNet** (cardiac phase detection) and **EchoLVSNet** (LV segmentation), with a disk-summation volume estimate.

---

## 📌 Overview
```
DICOM / video
   │
   ▼
EchoPDNet ── sliding-window phase curve ── ED / ES frames for every beat
   │
   ▼
EchoLVSNet ── LV mask on each ED and ES frame
   │
   ▼
Disk summation (method of discs, K = 20) ── EDV, ESV
   │
   ▼
EF = (EDV − ESV) / EDV × 100   per beat  →  mean / median EF
```

---

## 📥 Input
| Item | Details |
|------|---------|
| File | One A4C cine as a **DICOM** (`.dcm`) or a **video** (`.avi`, `.mp4`, … anything OpenCV can read) |
| Length | Any number of frames and beats (a video under 30 frames is zero-padded for the phase model) |
| Pixel spacing | **DICOM:** read from its metadata (`PixelSpacing`, or `PhysicalDeltaX` as fallback). **Video:** has none, so it is taken from a same-named `.dcm` next to it, else from the optional `pixel_spacing_mm` argument, else volumes are reported in px³ |
| Colour | Colour or YBR cines are converted to grayscale |

> Pixel spacing only affects the absolute volumes (mL vs px³). **EF is a ratio and does not depend on it.**

## 📤 Output
`compute_ef()` returns a dictionary (and, with `output_dir`, saves it as `result.json` with extra files):

| Field | Meaning |
|-------|---------|
| `ED_frames`, `ES_frames` | All detected end-diastolic / end-systolic frame indices (0-based) |
| `beats` | One entry per ED→ES beat: `ED_frame`, `ES_frame`, `EDV_<unit>`, `ESV_<unit>`, `EF_percent`, `valid` |
| `EF_mean`, `EF_median` | EF in %, over the beats with `EDV > ESV > 0` |
| `unit` | `"mL"` if pixel spacing was available, else `"px^3"` |
| `num_frames`, `input` | Bookkeeping |

All values are rounded to 2 decimals. With `output_dir` set (the folder is emptied first), it also writes `frame_XXXX.png` and `frame_XXXX_mask.png` for every ED/ES frame used, and `phase_curve.txt` (one phase value per frame).

---

## 🚀 Usage
```python
from echoforge.ef_pipeline import compute_ef

result = compute_ef("study.dcm", output_dir="ef_output")      # DICOM
print(result["EF_mean"], result["beats"])

result = compute_ef("clip.avi", pixel_spacing_mm=0.4)          # video; spacing optional
```

To process many files, load the models once and reuse them:
```python
from echoforge import load_model
from echoforge.ef_pipeline import compute_ef

pd_model = load_model("EchoPDNet", pretrained=True)
seg_model = load_model("EchoLVSNet", pretrained=True)
for path in paths:
    compute_ef(path, pd_model=pd_model, seg_model=seg_model)
```

---

## 🧩 What the pipeline uses
| Stage | Component | Notes |
|-------|-----------|-------|
| Phase detection | [**EchoPDNet**](../phasedetection/models/echopdnet/README.md) (ResNet50 + 2×LSTM) | Frames resized to 112×112 (3-channel), scaled to [0, 1]; windows of 30 frames, stride 1, overlapping predictions averaged |
| Beat detection | `scipy.signal.find_peaks` on the min-max normalised phase curve | `distance=10`, `prominence=0.05`; maxima = ED, minima = ES; each ED is paired with the first ES before the next ED |
| Segmentation | [**EchoLVSNet**](../segmentation/README.md) (U-Net) | Frame padded to 512×512 keeping aspect ratio, **raw 0–255 input** (the model rescales internally), threshold 0.5, mask mapped back to the original frame size |
| Volume | `volume.py` (disk summation) | Long axis from apex to mitral-annulus midpoint cut into K = 20 discs; sum of π·r²·h, times `spacing_cm³` |
| DICOM / video I/O | `pydicom`, `OpenCV` | |

Adapted from the in-house `maskvolume.py` (P. Naidoo, 2022). `volume.py` works for any mask size (the original helper hard-coded a 512×512 canvas and cropped larger masks).

## 📁 Files
| File | Purpose |
|------|---------|
| `pipeline.py` | `compute_ef()` and the steps above (I/O, phase curve, beat detection, segmentation, EF) |
| `volume.py` | Disk-summation volume of a binary mask |
| `__init__.py` | Exports `compute_ef` |

## 📦 Extra dependencies
`opencv-python`, `scipy`, `pydicom`, `scikit-image` (in addition to the EchoForge requirements).

---

## ⚠️ Limitations
- Assumes an **A4C** view. There is no view or image-quality check; use the classification module first if needed.
- Single-plane disk summation is an approximation and is not equivalent to biplane Simpson's volumes; absolute volumes depend on the correct pixel spacing.
- Beat detection quality directly drives the result. A short false ED→ES pair (for example at the end of a clip) is averaged in; the median in `EF_median` is more robust. `MIN_GAP` in `pipeline.py` (default 0 = off) can be raised to ignore ES frames too close after an ED.
- Not yet validated against reference EF measurements. Treat as a research tool, not for clinical use.

## 🧪 Example run (illustrative)
Two de-identified A4C DICOMs from the in-house set:

| Clip | Frames | ED / ES | EF per beat (%) | EF mean (%) |
|------|--------|---------|-----------------|-------------|
| A | 48 | 1, 25 / 11, 35 | 47.50, 40.54 | 44.02 |
| B | 82 | 2, 65 / 40, 68 | 46.60, 1.35 (3-frame false beat) | 23.98 (EF_median 23.98; first beat alone 46.60) |

---

## 🔍 Explore Other Modules in EchoForge
- [Phase Detection](../phasedetection/README.md)
- [Segmentation](../segmentation/README.md)
- [Classification](../classification/README.md)

## 📜 License
CC BY-NC-SA 4.0. See the [LICENSE](../license/LICENSE.txt) file for details.

---

_This is part of the THRIVE Research Centre [view EchoForge](../../README.md) project ecosystem._
