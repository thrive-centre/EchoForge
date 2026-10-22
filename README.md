<table width="100%">
  <tr>
    <td width="20%" align="left">
      <img src="echoforge/images/thrive_logo.jpg" alt="Thrive Intelligent Health" width="140">
    </td>
    <td width="60%" align="center">
      <h1>EchoForge</h1>
      <b>Pretrained deep learning models for echocardiography</b>
    </td>
    <td width="20%"></td>
  </tr>
</table>
**EchoForge** is a modular deep learning library built for echocardiographic image analysis. It provides seamless access to pretrained models for classification and segmentation, and will expand to include landmark detection, image quality assessment, and more.

The library is built using TensorFlow/Keras and integrates with models hosted on Hugging Face. It is designed for researchers, clinicians, and ML developers working with cardiac ultrasound imaging.


**From View Classification to Segmentation and Timing Detection — All in One Library**

<!-- <p align="center">
  <img src="echoforge/images/calssifcation_view.mp4" width="32%" alt="View Classification"/>
  <img src="echoforge/images/segmentaion.gif" width="32%" alt="LV Segmentation"/>
  <img src="echoforge/images/phase_detection_banner.png" width="32%" alt="Phase Detection"/>
</p> -->

### From View Classification to Segmentation and Timing Detection

**1. View Classification**

<p align="center">
  <video width="85%" autoplay loop muted playsinline>
    <source src="echoforge/images/calssifcation_view.mp4" type="video/mp4">
  </video>
</p>

**2. Left Ventricular Segmentation**

<p align="center">
  <img src="echoforge/images/segmentaion.gif" width="65%" alt="Left Ventricular Segmentation"/>
</p>

**3. Cardiac Phase Detection**

<p align="center">
  <video width="85%" autoplay loop muted playsinline>
    <source src="echoforge/images/phase.mp4" type="video/mp4">
  </video>
</p>

<p align="center">
  <em>Illustrative graphics (not real patient data).</em>
</p>

---

## 🔧 Features

- Pretrained model access (Hugging Face integration)
- Modular and scalable model zoo
- Supports both private and public Hugging Face repositories
- Central `load_model()` interface for dynamic loading
- Architecture-ready for classification, segmentation, landmarking, and timing estimation
- Caches downloaded models locally to avoid repeated downloads

---
<!-- 
## 📦 Installation

```bash
git clone https://github.com/intsav/EchoForge.git
cd EchoForge
pip install -e .
``` -->
## 📦 Installation

EchoForge can be installed in two ways depending on your workflow:  
a standard local installation, or installation inside a TensorFlow Docker/Jupyter environment.

---

### 🔧 Standard Installation (Local / Virtual Environment)

Use this option if you are installing EchoForge on your own machine or in a Python virtual environment.

```bash
git clone https://github.com/thrive-centre/EchoForge.git
cd EchoForge
pip install -e .
```
This installs EchoForge in editable mode, allowing you to update the source code and test changes instantly.

## 🐋 Installation Inside a TensorFlow Docker (GPU + Jupyter)

If you're running EchoForge inside a TensorFlow Docker container such as:

```bash
tensorflow/tensorflow:2.13.0-gpu-jupyter
```

then install it using the following commands inside your notebook or container terminal:

```bash
# Clone the EchoForge repository
!git clone https://github.com/thrive-centre/EchoForge.git

# Install EchoForge from the mounted directory
!pip install -e /tf/notebooks/EchoForge
```

>Note:
Update the path /tf/notebooks/EchoForge if your directory structure differs.
This is simply where your repo is located inside the container.
---

## Importing The Model 

```python
from echoforge import load_model

# Load any model by name from the registry
model = load_model("EchoFine_classifier", pretrained=True)
```
---

## 🧾 Model Zoo

| Task | Model | Docs |
|---|---|---|
| View classification | EchoFine (classifier, contrastive encoder, TMED2), EchoSDNet | [Classification](echoforge/classification/README.md) · [EchoFine](echoforge/classification/models/EchoFine/README.md) · [EchoSDNet](echoforge/classification/models/EchoSDNet/README.md) |
| LV segmentation | EchoLVSNet | [Segmentation](echoforge/segmentation/README.md) · [EchoLVSNet](echoforge/segmentation/models/echolvsnet/README.md) |
| Phase detection (ED/ES) | EchoPDNet | [Phase Detection](echoforge/phasedetection/README.md) · [EchoPDNet](echoforge/phasedetection/models/echopdnet/README.md) |

Weights, datasets and model details are available through [THRIVE](https://www.thrive-centre.com), including the [datasets page](https://www.thrive-centre.com/datasets) (e.g. [TTE47](https://www.thrive-centre.com/datasets/TTE47)). Registered model names are in `echoforge/config/registry.json`, or call `list_models()`.

```python
from echoforge import load_model, list_models

print(list_models())
model = load_model("EchoLVSNet")   # swap the name to load any model
# e.g. "EchoFine_classifier_ConvNeXt-T", "ViT_TTE47_baseline", "EchoFine_TMED2_fold0"
```

### 📚 Modules
- [Classification](echoforge/classification/README.md)
- [Segmentation](echoforge/segmentation/README.md)
- [Phase Detection](echoforge/phasedetection/README.md)

### 🔗 THRIVE Resources
- 🏠 [THRIVE Centre](https://www.thrive-centre.com)
- 🗂️ [Datasets](https://www.thrive-centre.com/datasets) (e.g. [TTE47](https://www.thrive-centre.com/datasets/TTE47), [UnityLV-MultiX](https://www.thrive-centre.com/datasets/UnityLV-MultiX))
- 🤗 [THRIVE on Hugging Face](https://huggingface.co/thrive-2025)

> Registered models are listed in `echoforge/config/registry.json`. Private repos need the `HF_TOKEN` environment variable.

---

## 👥 Credits

- Developed by: **THRIVE Centre**
- Research Group: **Intelligent Sensing and Vision**
- Clinical Collaboration: **Faculty of Medicine, Imperial College London**

---

## 📎 Notes

- All models are downloaded once and cached in `~/.echoforge/models/`
- You can load models without pretrained weights for retraining
