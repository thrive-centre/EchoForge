# Segmentation Module

> **EchoForge Segmentation** brings together deep learning models designed to segment anatomical structures, regions, or pathologies from echocardiographic images. This module serves as a central hub for exploring and accessing segmentation tools developed by THRIVE Research Centre.

---

## 📌 Overview
Echocardiographic segmentation is critical for:
- **Quantitative cardiac assessment**
- **Region-specific tissue tracking**
- **Automated measurement of clinical parameters** (e.g., LV volume, wall thickness)

The models in this module are trained to segment structures such as:
- **Left Ventricle (LV)**


Each model is tailored for specific tasks, views, and datasets.

---

## 📝 Tasks Covered 
| Task | Description |
|------|-------------|
| LV Segmentation | Binary segmentation of the left ventricle from 2CH/4CH views |
| Multi-Class Segmentation| Segmentation of left and right ventricle, Myocardium, etc |


## 📊 Datasets Used

The following datasets have been used across various models in this module:

### 🏥 1. **EchoNet-Dynamic**
- Public dataset with 10,000+ labeled apical 4CH videos
- Frame-level LV segmentation masks
- [Link to dataset](https://echonet.github.io/dynamic/)

### 🏥 2. **CAMUS**
- Contains 2CH and 4CH echocardiograms with LV/LA segmentation
- Annotated by cardiologists
- [Link to dataset](https://www.creatis.insa-lyon.fr/Challenge/camus/)

### 🏥 3. **HMC-QU**
- Echocardiography dataset collected at Hamad Medical Corporation and Qatar University
- Includes annotated 2D echo frames for segmentation tasks (e.g., LV cavity, myocardium)
- [Link to dataset](https://www.kaggle.com/datasets/aysendegerli/hmcqu-dataset)

### 🏥 4. **UNITY**
- Proprietary dataset annotated by THRIVE Research Centre
- Used for specific internal benchmarking

> For detailed preprocessing steps and dataset-specific splits, see the individual model documentation.

---

## 🟢 Available 

Below is the list of segmentation models included in this module:

| Model Name     | Task             | Architecture | Dataset           | Dice Score | IoU Score | Link                                                 |
|----------------|------------------|--------------|-------------------|------------|-----------|------------------------------------------------------|
| **EchoLVSNet** | LV Segmentation  | U-Net        | EchoNet-Dynamic   | 0.91       | 0.85      | [View Model ➜](models/echolvsnet/README.md)               |
|                |                  |              | CAMUS             | 0.94        | 0.88       | [View Model ➜](models/echolvsnet/README.md)               |
|                |                  |              | HMC-QU            | –          | –         | [View Model ➜](models/echolvsnet/README.md)               |
|                |                  |              | UNITY             | 0.90         | 0.83        | [View Model ➜](models/echolvsnet/README.md)               |


---

## 🔍 Explore Other Modules in EchoForge:
- [Classification](../classification/README.md)
- [Phase Detection](../phasedetection/README.md)


## 📜 License
CC BY-NC-SA 4.0 
See the [LICENSE](../license/LICENSE.txt) file for details.


---

_This is part of the THRIVE Research Centre [view EchoForge](../../README.md) project ecosystem._


