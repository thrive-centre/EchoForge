# Phase Detection Module

> **EchoForge Phase Detection** delivers cutting-edge deep learning models tailored to detect key cardiac phases—specifically End-Diastole (ED) and End-Systole (ES)—from echocardiographic video sequences. This module supports temporal modelling for cardiac cycle interpretation and benchmarking against expert cardiologist annotations.

---

## 📌 Overview
Accurate cardiac phase detection is essential for:
- **Quantifying heart function (e.g., ejection fraction)**
- **Automated selection of clinically relevant frames (ED/ES)**
- **Standardised downstream measurements and segmentation**

The models in this module are optimised for:
- **Frame-wise phase regression**
- **Multibeat echocardiographic sequences**
- **Evaluation on both clinical and public datasets**

---

## 📝 Tasks Covered 
| Task | Description |
|------|-------------|
| ED/ES Frame Detection| Predict the location of End-Diastolic and End-Systolic frames from temporal sequences|
| Continuous Phase Regression| Output smooth phase curves for multi-beat sequences|

---

## 📊 Datasets Used
The following datasets are used for training and external evaluation of phase detection models:

### 🏥 1. **PACS Dataset**
- Clinical echocardiographic video dataset from Imperial College Healthcare NHS Trust
- 1,000 A4C clips annotated by accredited cardiologists
- 60/20/20 split for training/validation/test

### 🏥 2. **MultiBeat Dataset**
- External benchmark dataset with multiple heartbeats
- Blind-tested by the model to evaluate generalisability

### 🏥 3. **EchoNet-Dynamic**
- Public dataset with 10,000+ A4C videos
- Used for testing only; no fine-tuning performed
- [Link to dataset](https://echonet.github.io/dynamic/)

> For details on dataset preparation and annotation protocols, see the model documentation.

---

## 🟢 Available
Below is the list of phase detection models currently supported in this module:


| Model Name   | Task                 | Architecture         | Dataset          | ED aaFD | ES aaFD | Link         |
|--------------|----------------------|-----------------------|------------------|---------|---------|--------------|
| **EchoPDNet**| ED/ES Frame Regression | ResNet50 + 2x-LSTM  | PACS             | 0.66    | 0.81    | [View Model ➜](../models/echopdnet/README.md)        |
|              |                      |                       | MultiBeat        | 2.62    | 1.86    | [View Model ➜](../models/echopdnet/README.md)        |
|              |                      |                       | EchoNet-Dynamic  | 2.30    | 3.49    | [View Model ➜](../models/echopdnet/README.md)        |

---

## 🔍 Explore Other Modules in EchoForge:
- [Segmentation](../segmentation/README.md)
- [Classification](../classification/README.md)

---

## 📜 License

CC BY-NC-SA 4.0  
See the [LICENSE](../license/LICENSE.txt) file for details.

---

_This is part of the THRIVE Research Centre [view EchoForge](../README.md) project ecosystem._