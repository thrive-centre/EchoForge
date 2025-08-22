# 🔄 EchoPDNet – Multibeat Echocardiographic Phase Detection

**EchoPDNet** is a deep learning model designed to automatically detect **End-Diastolic (ED)** and **End-Systolic (ES)** frames from echocardiographic video sequences. It employs a **spatiotemporal architecture** that combines spatial features extracted via a ResNet50 encoder with temporal dependencies captured using stacked LSTM layers.

---

## 🧠 Model Overview

- **Architecture:** ResNet50 (ImageNet) + 2x LSTM
- **Input Shape:** `(30, 112, 112, 3)` video segment (30 frames)
- **Output:** Frame-wise **continuous phase score** (regression)
- **Loss Function:** Mean Squared Error (MSE)
- **Metrics:** Mean Absolute Error (MAE), Average Absolute Frame Difference (aaFD)
- **Framework:** Keras Functional API

---

## 📦 Dataset Summary

### 🏥 1. **PACS Dataset** (Training/Validation/Test)
- 1,000 A4C echocardiography videos from Imperial College Healthcare NHS Trust
- Videos contain 1–3 heartbeats
- Ground-truth ED/ES frame annotations by accredited cardiologists
- **Split:** 60% train / 20% val / 20% test

### 🏥 2. **MultiBeat Dataset** (External Test Set)
- Unseen dataset for generalisability testing
- Used to benchmark detection reliability against inter-observer variability

### 🏥 3. **EchoNet-Dynamic** (External Public Test Set)
- 10,000+ A4C echo videos
- Used for blind evaluation of ED/ES detection performance

---

## 📈 Results

### ▶ PACS Dataset
| Phase | Avg Frame Diff (μ±σ) | aaFD |
|-------|-----------------------|------|
| **ED** | -0.09 ± 1.10 frames | 0.66 |
| **ES** |  0.11 ± 1.29 frames | 0.81 |

- Performance on par with human experts
- Inference time: ~1.5 seconds vs. 26±11s for clinicians

### ▶ MultiBeat Dataset
| Phase | Avg Frame Diff (μ±σ) | aaFD |
|-------|-----------------------|------|
| **ED** | -1.34 ± 3.27 frames | 2.62 |
| **ES** | -0.31 ± 3.37 frames | 1.86 |

### ▶ EchoNet-Dynamic (External)
| Phase | Avg Frame Diff (μ±σ) | aaFD |
|-------|-----------------------|------|
| **ED** |  0.16 ± 3.56 frames | 2.30 |
| **ES** |  2.64 ± 3.59 frames | 3.49 |

> All errors were within or better than inter-observer variability.

---

## 🔧 Model Summary

```python
from echoforge import load_model
model = load_model("EchoPDNet", pretrained=True)
```

- Uses ResNet50 encoder to extract spatial features from each frame
- Applies 2-layer LSTM to sequence of features
- Outputs a 30-length vector representing the **regressed phase scores**
- Peaks (maxima = ED, minima = ES) are extracted using a **peak detection** algorithm

---

## ⚙️ Architecture
| Layer                        | Expected Input Shape | Description                                                                             |
| ---------------------------- | -------------------- | --------------------------------------------------------------------------------------- |
| **Input Layer**              | `(30, 112, 112, 3)`  | A sequence of 30 RGB echo frames (A4C view), resized to 112×112 pixels.                 |
| **TimeDistributed ResNet50** | `(30, 112, 112, 3)`  | Applies ResNet50 to each frame individually, producing a `(30, 2048)` feature sequence. |
| **LSTM Layer 1**             | `(30, 2048)`         | Receives temporal sequence of ResNet-encoded features across 30 frames.                 |
| **Dense (ReLU)**             | `(30, 2048)`         | Projects high-dimensional features to a reduced dimension of 512.                       |
| **LSTM Layer 2**             | `(30, 512)`          | Further models temporal dependencies in the cardiac cycle.                              |
| **Flatten**                  | `(30, 512)`          | Flattens the sequence for final regression.                                             |
| **Dense Output**             | `15360` → `(30,)`    | Produces a 30-element vector of **continuous phase scores**, one per frame.             |


---

## 🗂 Source

- 📝 Paper: *Multibeat Echocardiographic Phase Detection Using Spatiotemporal Deep Learning*
- 🧪 Model Training: THRIVE Research Centre
- 📦 Dataset: PACS (internal), EchoNet-Dynamic (external)
