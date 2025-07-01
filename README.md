# EchoForge

**EchoForge** is a modular deep learning library built for echocardiographic image analysis. It provides seamless access to pretrained models for classification and will support segmentation, landmark detection, and more in upcoming releases.

The library is built using TensorFlow/Keras and integrates with models hosted on Hugging Face. It is designed for researchers, clinicians, and ML developers working with cardiac ultrasound imaging.

---

## 🔧 Features

- Pretrained model access (Hugging Face integration)
- Modular and scalable model zoo
- Support for private and public Hugging Face repositories
- Central `load_model()` interface
- Future-proof architecture for segmentation, landmarking, timing, and quality assessment

---

## 📦 Installation

```bash
git clone https://github.com/intsav/EchoForge.git
cd echoforge
pip install -e .
```

---

## 🧠 Usage

### Option 1: Generic Loader

```python
from echoforge import load_model

model = load_model("Echo2DClassifier", pretrained=True)
```

### Option 2: Direct Import

```python
from echoforge.models import Echo2DClassifier

model = Echo2DClassifier(pretrained=True)
```

---

## 📂 Available Models

| Model Name         | Task                   | Status |
|--------------------|------------------------|--------|
| [Echo2DClassifier](./models/echo2dclassifier/README.md)   | 2D View Classification | ✅ Released |
| EchoSegNet         | Chamber Segmentation   | 🔜 Coming soon |
| EchoLandmarker     | Landmark Detection     | 🔜 Coming soon |

---

## 🚧 Roadmap

- [x] View classification (Echo2DClassifier)
- [ ] Segmentation model (EchoSegNet)
- [ ] Landmark detection (EchoLandmarker)
- [ ] Timing estimation & quality prediction
- [ ] PyPI package release

---

## 👥 Credits

- Developed by: **Mayur Agrawal**
- Collaborators: XXXXX
