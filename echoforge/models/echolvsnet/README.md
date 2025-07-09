# 🫀 Left Ventricular Segmentation on EchoNet-Dynamic with U-Net

This project implements a robust deep learning segmentation model **EchoLVSNet** to perform **segmentation of the left ventricle** on the [EchoNet-Dynamic dataset](https://echonet.github.io/dynamic/). It uses a **U-Net architecture** built in TensorFlow and supports **transfer learning** from a pre-trained model.

---
## 🔁 Transfer Learning

This project supports loading a pre-trained model developed by one of the THRIVE PhD students. The model was trained on a similar left ventricular segmentation task on the UNITY Dataset. This projects fine-tunes the segmentation model on the EchoNet-Dynamic dataset.

---

## 📊 Dataset

This project uses the **EchoNet-Dynamic dataset**, a large-scale echocardiography dataset released by the Stanford ML Group.

- **Modality:** Transthoracic echocardiography (TTE)
- **View:** Apical 4-Chamber (A4C)
- **Input Format:** 512x512 grayscale PNG frames
- **Labels:** Corresponding binary segmentation masks of the **left ventricle**
- **Preprocessing:** Extracted video frames and masks, resized and normalised

### 📁 Dataset Split

| Set        | Number of Samples |
|------------|-------------------|
| Train      | 15,048            |
| Validation | 3,000             |
| Test       | 1,998             |


---

## 🧠 Model Overview

- **Architecture:** U-Net (with encoder-decoder structure and skip connections)
- **Input Shape:** 512 x 512 grayscale PNG frames
- **Output:** Predicted segmentation mask of the left ventricle
- **Loss Function:** Binary Crossentropy + Dice Loss
- **Metrics:** Dice Coefficient, IoU & pixel-wise accuracy

---

## 📈 Results - Test Set

- **Mean Dice Coefficient:** 0.92
- **IoU score:** 0.85
- **Pixel-wise Accuracy:** 0.98

---

## 📥 How to Use

```python
from echoforge import load_model
model = load_model("EchoLVSNet", pretrained=True)
```

Or load only the architecture:

```python
from echoforge.models import EchoLVSNet
model = EchoLVSNet(pretrained=False)
```

---

## 🔍 Model Summary

```
Total model:   31125921 parameters
Trainable params: 31109921 parameters
Non-trainable params: 16000 parameters
```

---

## 📤 Source

- Dataset: EchoNet-Dynamic - https://echonet.github.io/dynamic/
- Training by: Arshian Hussain, THRIVE Research Centre
