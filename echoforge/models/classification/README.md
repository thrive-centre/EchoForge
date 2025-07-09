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
| Binary Disease Classification | Classify presence/absence of specific cardiac pathology |
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

| Model Name           | Task                         | Architecture     | Accuracy | F1 Score | Link |
|----------------------|------------------------------|------------------|----------|----------|------|
| **Echo2DClassifier** | 2D View Classification        | ResNet50    | 94.2%    | 0.92     | [View Model ➜](../echo2dclassifier/README.md) |
| **Coming Soon**      | Multi-Class Classification         | —                | —        | —        | [View Model ➜](../echo2dclassifier/README.md) |

---

## 🔍 Explore Other Modules in EchoForge:
- [Segmentation](../segmentation/README.md)
- [Phase Detection](../phase-detection/README.md)
- [View Recognition](../view-recognition/README.md)

---

## 📜 License

CC BY-NC-SA 4.0  
See the [LICENSE](../../license/LICENSE.txt) file for details.

---

_This is part of the THRIVE Research Centre [view EchoForge](https://github.com/EchoForge) project ecosystem._