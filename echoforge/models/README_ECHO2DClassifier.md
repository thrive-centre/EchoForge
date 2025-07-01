# Echo2DClassifier

**Echo2DClassifier** is the first official model in the EchoForge library. It is a robust deep learning model trained on the UNITY-Classification-B dataset to perform multi-class 2D echocardiographic view classification.

---

## 📌 Model Details

- **Architecture**: Xception (base) + Flatten + Dense(47)
- **Input Size**: 224 x 224 x 3
- **Output**: One of 47 standard echocardiographic views
- **Training**: Trained on high-quality labels from UNITY-Classification-B
- **Saved Format**: Full `.h5` model (architecture + weights)

---

## ✅ Classes Covered

47 standard echo views including:

```
a2ch-la, a2ch-lv, a2ch-full, a3ch-lv, a3ch-full, a3ch-la', a3ch-outflow,
a4ch-ias, a4ch-la, a4ch-lv, a4ch-rv, a4ch-ra, a4ch-full, a5ch-full, a5ch-outflow,
mmode-a4ch-rv, mmode-ivc, mmode-plax-mitral, mmode-plax-av, mmode-plax-lv,
apex, psax-lv-mid, psax-lv-base, psax-tv, psax-all, psax-av, psax-lv-apex, psax-pv,
plax-valves-av, plax-full-mv, plax-full-la, plax-full-out, plax-tv, plax-full-lv,
plax-valves-mv, plax-full-rv-ao, suprasternal, subcostal-ivc, subcostal-heart,
doppler-ao-descending, doppler-tissue-lateral, doppler-mv, doppler-av,
doppler-tissue-septal, doppler-pv, doppler-tissue-rv, doppler-tv
```

---

## 📥 How to Use

```python
from echoforge import load_model
model = load_model("Echo2DClassifier", pretrained=True)
```

Or load only the architecture:

```python
from echoforge.models import Echo2DClassifier
model = Echo2DClassifier(pretrained=False)
```

---

## 🔍 Model Summary

```
Xception base: 20.8M parameters
Total model:   25.5M parameters
```

---

## 📤 Source & Training

- Dataset: UNITY-Classification-B
- Labels by: Darrel
- Training by: Mayur Agrawal
