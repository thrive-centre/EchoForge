# EchoForge

**EchoForge** is a modular, lightweight deep learning library for echocardiographic image analysis. It provides plug-and-play access to pretrained models for classification, with support for future tasks like segmentation, landmark detection, and motion analysis.

This project is built on TensorFlow and integrates seamlessly with models hosted on Hugging Face. It is designed for both research and production use.

---

## 🔧 Features

- ✅ Pretrained model loading from Hugging Face
- ✅ Easy toggle between pretrained and architecture-only
- ✅ Built-in support for private and public models
- ✅ Central `load_model()` utility + direct `Echo2DClassifier()` access
- 📚 Designed for extension to multiple model types
- 🧪 Includes a model verification test script

---

## 🫀 Included Model

### `Echo2DClassifier`
- **Task**: 2D View Classification  
- **Architecture**: `Xception + Flatten + Dense(47)`  
- **Input Size**: `224 x 224 x 3`  
- **Output Classes**: 47  
- **Training Data**: UNITY-Classification-B  
- **Weights**: Hugging Face (full `.h5` model)

---

## 📦 Installation

```bash
git clone https://github.com/intsav/EchoForge.git
cd echoforge
pip install -e .
```

---

## 🔐 Setup for Private Hugging Face Models

To access private models hosted on Hugging Face:

1. [Create a Hugging Face account](https://huggingface.co)
2. [Generate a token](https://huggingface.co/settings/tokens) with **read access**
3. Create a `.env` file in your repo root:

```env
HF_TOKEN=hf_your_token_here
```

> ⚠️ Do **not** commit `.env` to GitHub. It is already `.gitignore`d.

---

## 🧠 Usage

### Option A: Use `load_model()` (Recommended)

```python
from echoforge import load_model

# Load pretrained model from Hugging Face
model = load_model("Echo2DClassifier", pretrained=True)

# Or load just the architecture (ImageNet base)
model = load_model("Echo2DClassifier", pretrained=False)

model.summary()
```

### Option B: Use model directly

```python
from echoforge.models import Echo2DClassifier

model = Echo2DClassifier(pretrained=True)
```

---

## 🧾 Echo2DClassifier Output Classes

| Class Names |
|-------------|
| a2ch-la, a2ch-lv, a2ch-full, a3ch-lv, a3ch-full, a3ch-la', a3ch-outflow |
| a4ch-ias, a4ch-la, a4ch-lv, a4ch-rv, a4ch-ra, a4ch-full, a5ch-full, a5ch-outflow |
| mmode-a4ch-rv, mmode-ivc, mmode-plax-mitral, mmode-plax-av, mmode-plax-lv |
| apex, psax-lv-mid, psax-lv-base, psax-tv, psax-all, psax-av, psax-lv-apex, psax-pv |
| plax-valves-av, plax-full-mv, plax-full-la, plax-full-out, plax-tv, plax-full-lv |
| plax-valves-mv, plax-full-rv-ao, suprasternal, subcostal-ivc, subcostal-heart |
| doppler-ao-descending, doppler-tissue-lateral, doppler-mv, doppler-av |
| doppler-tissue-septal, doppler-pv, doppler-tissue-rv, doppler-tv |

---

## 🧪 Test Script

Run the test script to verify model architecture, download, and weight comparison:

```bash
python test.py
```

This will:
- Load both pretrained and architecture-only versions
- Compare weights to confirm training status

---

## 📈 Roadmap

- [x] Echo2DClassifier (47 classes)
- [ ] Segmentation models (multi-chamber, LV/LA)
- [ ] Landmark detection models
- [ ] Motion/timing estimation
- [ ] PyPI release (`pip install echoforge`)

---

## 👥 Contributors & Notes

- Dataset: UNITY-Classification-B  
- Labels by: **XXX**  
- Lead Developer: **XXX**

---

## ⚖️ License

MIT License — free for academic and non-commercial use.

---

## 🤝 Contributing

We welcome contributions! You can:
1. Fork this repo
2. Add your model or feature
3. Submit a pull request

Please include documentation and tests with your contribution.
