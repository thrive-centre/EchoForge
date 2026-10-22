<h1 align="center">🫀 EchoForge</h1>

<p align="center"><b>Pretrained deep learning models for echocardiography</b><br>TensorFlow/Keras · Hugging Face · THRIVE Centre</p>

**EchoForge** is a modular deep learning library built for echocardiographic image analysis. It provides seamless access to pretrained models for classification and segmentation, and will expand to include landmark detection, image quality assessment, and more.

The library is built using TensorFlow/Keras and integrates with models hosted on Hugging Face. It is designed for researchers, clinicians, and ML developers working with cardiac ultrasound imaging.


**From View Classification to Segmentation and Timing Detection — All in One Library**

#### View Classification  
![Echocardiographic View Classification](echoforge/images/classificaiton.png)

#### LV Segmentation  
![Chamber Segmentation in Echo Images](echoforge/images/segmentaion.gif)

#### Phase Detection (ED vs ES Labelling)  
![Phase Detection](echoforge/images/pacs-examples.png)


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
model = load_model("EchoView47_classifier", pretrained=True)
```
---

## 🧾 Model Zoo

| Model | Task | `load_model` name | Docs | Weights (Hugging Face) | Datasets |
|---|---|---|---|---|---|
| **EchoView47 Classifier** | 47-class view classification | `EchoView47_classifier` | [README](echoforge/classification/models/EchoView47/README.md) | [thrive-2025/EchoView47](https://huggingface.co/thrive-2025/EchoView47) | [TTE47](https://www.thrive-centre.com/datasets/TTE47) |
| **EchoView47 Contrastive Encoder** | Self-supervised view encoder | `EchoView47_contrastive_encoder` | [README](echoforge/classification/models/EchoView47/README.md) | [thrive-2025/EchoView47](https://huggingface.co/thrive-2025/EchoView47) | [TTE47](https://www.thrive-centre.com/datasets/TTE47) |
| **EchoView47 TMED2** | 5-class TMED2 view classification | `EchoView47_TMED2` | [README](echoforge/classification/models/EchoView47/README.md) | [thrive-2025/test_echoforge](https://huggingface.co/thrive-2025/test_echoforge) | TMED2 |
| **EchoSDNet** | Single vs. dual-view classification | `EchoSDNet` | [README](echoforge/classification/models/EchoSDNet/README.md) | [thrive-2025/EchoSDNet](https://huggingface.co/thrive-2025/EchoSDNet) | UNITY |
| **EchoLVSNet** | LV segmentation | `EchoLVSNet` | [README](echoforge/segmentation/models/echolvsnet/README.md) | [thrive-2025/echolvsnet](https://huggingface.co/thrive-2025/echolvsnet) | UNITY |
| **EchoPDNet** | ED/ES phase detection (docs only) | _coming soon_ | [README](echoforge/phasedetection/models/echopdnet/README.md) | – | – |

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
- Future versions will include `torch` support and ONNX export