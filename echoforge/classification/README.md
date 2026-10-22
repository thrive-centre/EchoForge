# Classification Module

> **EchoForge Classification** provides deep learning models for echocardiographic view classification, cardiac phase detection, etc. This module serves as a central hub for all classification tasks developed by the THRIVE Research Centre.

---

## 📌 Overview

Echocardiographic classification models are essential for:
- **Standardising view identification** (e.g., A4C, A2C, PLAX)
- **Filtering poor-quality or misclassified views**
- **Detecting phases or pathology from still frames or video**

The models in this module cover tasks such as:
- **View classification**
- **Phase detection (systole/diastole)**
- **Binary/multi-label cardiac condition classification**

## 📝 Tasks Covered

| Task                         | Description |
|------------------------------|-------------|
| View Classification          | Identify the anatomical view (e.g., A4C, PLAX) from 2D echo |
| Phase Classification         | Classify frames or sequences into systole/diastole |
| Multi-class Classification   | Predict multiple classes or labels simultaneously |

---

## 📊 Datasets Used

The following datasets have been used across classification models:

### 🏥 1. **T-MED**
- A curated clinical dataset featuring multi-view and phase-labelled echocardiograms
- Useful for both view classification and disease prediction tasks
- Annotated by medical experts from collaborative teaching hospitals
- Access on request

### 🏥 2. **UNITY**
- Internal dataset for multi-class pathology classification
- Manually curated and annotated by THRIVE Research Centre

> For view-specific or condition-specific filtering criteria, refer to individual model documentation.

---

## 🟢 Available 

Below is the list of classification models included in this module:

| Model Name     | Task                   | Architecture             | Dataset | Accuracy | F1-Score | Link                                                  |
|----------------|------------------------|--------------------------|---------|----------|----------|-------------------------------------------------------|
| **EchoFine** | 2D View Classification | ResNet50                 | UNITY   | 94.1%    | 0.92     | [View Model ➜](models/EchoFine/README.md)       |
|                |                        |                          | T-MED   | 98.03%   | –        | [View Model ➜](models/EchoFine/README.md)       |
| **EchoSDNet**  | Binary Classification  | ResNet50 + Custom Layers | UNITY   | 99.48%   | 0.99     | [View Model ➜](models/EchoSDNet/README.md)        |



### 🤗 Load any model by name

```python
from echoforge import load_model
model = load_model("EchoFine_classifier")
```

**EchoFine (UNITY, 47 views)**

| `load_model` name | Description |
|---|---|
| `EchoFine_classifier` | 47-class view classifier |
| `EchoFine_contrastive_encoder` | Contrastive pretrained encoder |
| `EchoFine_TMED2` | Fine-tuned on TMED2 |
| `EchoFine_TMED2_fold0` / `_fold1` / `_fold2` | TMED2 cross-validation folds |

**Additional classifiers** (`EchoFine_classifier_<backbone>`) and **contrastive encoders** (`EchoFine_contrastive_encoder_<backbone>`), backbones: `ConvNeXt-T`, `EfficientNetV2`, `SwinTransV2-T`.

**Baselines without contrastive pretraining** (`<backbone>_TTE47_baseline`): `Xception`, `ResNet50`, `ResNet101`, `DenseNet121`, `EfficientNetV2`, `ConvNeXt`, `ViT`, `SwinTransformerV2`.

Run `from echoforge import list_models; print(list_models())` for the full list.

---

### 🤗 Load any model by name

```python
from echoforge import load_model
model = load_model("EchoFine_classifier")
```

**EchoFine (UNITY, 47 views)**

| `load_model` name | Description |
|---|---|
| `EchoFine_classifier` | 47-class view classifier |
| `EchoFine_contrastive_encoder` | Contrastive pretrained encoder |
| `EchoFine_TMED2` | Fine-tuned on TMED2 |
| `EchoFine_TMED2_fold0` / `_fold1` / `_fold2` | TMED2 cross-validation folds |

**Additional classifiers** (`EchoFine_classifier_<backbone>`) and **contrastive encoders** (`EchoFine_contrastive_encoder_<backbone>`), backbones: `ConvNeXt-T`, `EfficientNetV2`, `SwinTransV2-T`.

**Baselines without contrastive pretraining** (`<backbone>_TTE47_baseline`): `Xception`, `ResNet50`, `ResNet101`, `DenseNet121`, `EfficientNetV2`, `ConvNeXt`, `ViT`, `SwinTransformerV2`.

Run `from echoforge import list_models; print(list_models())` for the full list.

---

## 🔍 Explore Other Modules in EchoForge:
- [Segmentation](../segmentation/README.md)
- [Phase Detection](../phasedetection/README.md)

---

## 📜 License

CC BY-NC-SA 4.0  
See the [LICENSE](../license/LICENSE.txt) file for details.

---

_This is part of the THRIVE Research Centre [view EchoForge](../../README.md) project ecosystem._
