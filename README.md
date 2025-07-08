# EchoForge

**EchoForge** is a modular deep learning library built for echocardiographic image analysis. It provides seamless access to pretrained models for classification and segmentation, and will expand to include landmark detection, image quality assessment, and more.

The library is built using TensorFlow/Keras and integrates with models hosted on Hugging Face. It is designed for researchers, clinicians, and ML developers working with cardiac ultrasound imaging.

---

## 🔧 Features

- Pretrained model access (Hugging Face integration)
- Modular and scalable model zoo
- Supports both private and public Hugging Face repositories
- Central `load_model()` interface for dynamic loading
- Architecture-ready for classification, segmentation, landmarking, and timing estimation
- Caches downloaded models locally to avoid repeated downloads

---

## 📦 Installation

```bash
git clone https://github.com/intsav/EchoForge.git
cd EchoForge
pip install -e .
```

---

## 🧠 Usage

### Option 1: Generic Loader

```python
from echoforge import load_model

# Load any model by name from the registry
model = load_model("Echo2DClassifier", pretrained=True)
```

### Option 2: Direct Import

```python
from echoforge.models import Echo2DClassifier, EchoLVSNet

model = Echo2DClassifier(pretrained=False)
# or
model = EchoLVSNet(pretrained=True)
```

---

## 📂 Available Models

| Model Name           | Task                    | Status       |
|----------------------|-------------------------|--------------|
| [Echo2DClassifier](./echoforge/models/echo2dclassifier/README.md) | 2D View Classification | ✅ Released |
| [EchoLVSNet](./echoforge/models/echolvsnet/README.md)            | LV Segmentation         | ✅ Released |
| EchoLandmarker       | Landmark Detection      | 🔜 Coming soon |

---

## 🚧 Roadmap

- [x] 2D View Classification (Echo2DClassifier)
- [x] Left Ventricular Segmentation (EchoLVSNet)
- [ ] Landmark Detection (EchoLandmarker)
- [ ] Timing Estimation & Quality Prediction
- [ ] Model Benchmarking Suite
- [ ] PyPI Package Release
- [ ] Interactive Web Demo

---

## 👥 Credits

- Developed by: XXXXX
- Research Group: **IntSaV, Thrive Research Center**
- Clinical Collaboration: XXXXX

---

## 📎 Notes

- All models are downloaded once and cached in `~/.echoforge/models/`
- You can load models without pretrained weights for retraining
- Future versions will include `torch` support and ONNX export
