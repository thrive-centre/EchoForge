# EchoView47 

**EchoView47** is a deep learning model designed for echocardiographic
view classification. It has been trained on the **TTE47 dataset**,
covering **47 echocardiographic views**, and achieves **94.1%
accuracy on the TTE47 test split**.

The model is available in two forms:
- **EchoView47 Contrastive Encoder** - a pretrained encoder built on
**Xception**, trained using a contrastive learning framework with LogSum
loss on TTE47.
- **EchoView47 Classifier** - a fine-tuned 47-class classifier obtained
by training the encoder further with **Cross-Entropy loss**.

With its strong performance, EchoView47 can serve as both:
- A **feature extractor** for downstream echocardiography tasks, and
- A **ready-to-use classifier** for automated view classification.

--- 
## EchoView47 Contrastive Encoder (Pretrained Encoder)

| Layer (type)              | Output Shape        | Param #   |
|----------------------------|---------------------|-----------|
| rescaling (Rescaling)      | (None, 224, 224, 3) | 0         |
| xception (Functional)      | (None, 7, 7, 2048)  | 20,861,480|
| global_average_pooling2d   | (None, 2048)        | 0         |

- Total: 20,861,480 (79.58 MB)
- Trainable: 20,806,952 (79.37 MB)
- Non-Trainable: 54,528 (213 KB)

### Layer Breakdown 

**Rescaling (Rescaling)**  
  - **Purpose:** Normalises pixel values (e.g., scaling from [0–255] to [0–1]) to stabilise training and ensure consistent input.  
  - **Input:** Raw echocardiographic image of shape **(224, 224, 3)** (RGB).  

**Xception (Functional)**  
  - **Purpose:** A pretrained deep CNN backbone that extracts hierarchical image features (edges, textures, and echocardiographic structures). It compresses spatial dimensions while expanding feature richness.  
  - **Input:** Normalised images from the rescaling layer, shape **(224, 224, 3)**.  
  - **Output:** Feature maps of shape **(7, 7, 2048)**.  

**Global Average Pooling (GlobalAveragePooling2D)**  
  - **Purpose:** Reduces the 2D feature maps into a **1D vector** by averaging each channel. This reduces dimensionality while retaining semantic information.  
  - **Input:** Feature maps of shape **(7, 7, 2048)**.  
  - **Output:** A compact feature representation of size **(2048, )**.  
---

## EchoView47 Classifier (Fine-Tuned 47-class Classifier)

| Layer (type)               | Output Shape        | Param #   |
|-----------------------------|---------------------|-----------|
| sequential_1 (Sequential)   | (None, 2048)        | 20,861,480|
| dense (Dense)               | (None, 47)          | 96,303    |

- **Total Parameters:** 20,957,783 (79.95 MB)  
- **Trainable Parameters:** 20,903,255 (79.74 MB)  
- **Non-Trainable Parameters:** 54,528 (213 KB)  

---

### Layer Breakdown  

**Sequential (sequential_1)**  
  - **Purpose:** This block contains the **pretrained Xception backbone + rescaling + global average pooling** from the encoder. It outputs a compact **2048-dimensional feature vector** for each input echocardiographic image.  
  - **Input:** Raw image input of shape **(224, 224, 3)**.  
  - **Output:** Feature embedding of shape **(2048, )**.  

 **Dense (Dense, 47 units)**  
  - **Purpose:** Fully connected classification layer. Maps the 2048-dimensional feature vector into **47 probability scores**, one for each echocardiographic view in the TTE47 dataset.
  - **Input:** Feature vector of shape **(2048, )**.  
  - **Output:** Probability distribution across **47 classes**.
    
--- 

## TMED2 Classifier (EchoView47 Fine-Tuned on TMED2)

Pretrained with our Contrastive Framework using the LogSum Loss on **TTE47** and fine-tuned on **TMED2 (DEV479 splits)** with Cross-Entropy Loss:  

We follow the official **DEV479 split** (train, validation, and test), which provides labelled training samples across five categories.  
- Performance is reported as the **mean classification accuracy** across the three predefined folds for the four echocardiographic views (excluding the ‘other’ category).  
- For compatibility with the model architecture used in this work, all images were resized to **224 × 224 × 3**.  

---

## TTE47 Dataset 
The **TTE47 dataset** is derived from a random sample of real-world echocardiographic studies collected at **Imperial College Healthcare NHS Trust**, comprising a total of **91,139 images**. Ethical approval for the study was granted by the **Health Research Authority** (IRAS identifier: 243023). Only studies with complete demographic information and without intravenous contrast were included.  

Each image was **manually annotated by a cardiologist (Expert 1)** using a web-based platform ([Unity Imaging](https://unityimaging.net/)) and assigned to one of **47 predefined echocardiographic view categories**. The dataset was split as follows, ensuring no patient or study overlap between sets:  

- **Training set:** 76,589 images  
- **Validation set:** 9,103 images  
- **Test set:** 5,447 images  
- **Unique studies represented:** 19,169  

### Multi-Expert Test Set Annotation  
Two additional cardiologists (Experts 2 and 3) independently annotated the **test set**. Both were blinded to the original labels and to each other’s annotations. Unlike Expert 1, who was required to assign exactly one label per image, Experts 2 and 3 were permitted to select **“not sure”** when appropriate.  

The test set was intentionally sized to balance **comprehensive evaluation across all 47 views** with the feasibility of detailed multi-expert annotation. In total, the released test subset contains **5,447 images with annotations from all three experts**.  

### Dataset Splits & Metadata  

To support reproducibility and consistency across experiments, two key JSON files are provided (available for download on the **'datasplits' directory under EchoView47**):  

- **`class_lookup.json`**  
  - Maps the classifier’s output indexes to the corresponding **echocardiographic view names**.  
  - Ensures that predicted class indices can be directly interpreted as specific echocardiographic categories.  

- **`data_split.json`**  
  - Defines the **official training, validation, and test splits** for the dataset.  
  - Guarantees that researchers use a consistent splitting scheme with no patient overlap, enabling fair comparisons across studies.
  - Allows anyone to take the released classifier and make predictions on the **official test split**, ensuring that results can be **reproduced and benchmarked reliably**.
 
### Licensing and Availability  
- The **test subset** is released under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license**.  
- Dataset release was approved by the **South Central – Oxford C Research Ethics Committee** (IRAS ID: 279328, REC reference: 20/SC/0386).  
- Due to data governance restrictions, the **training set cannot be made public**; however, **pre-trained and fine-tuned model weights** are provided for research use.  

### Link to Dataset

You can find more details about the dataset (including the gallery of classes, demographics, and splits) on the official dataset page: [TTE47 — Dataset Card & Reference Gallery (Thrive Centre)](https://www.thrive-centre.com/datasets/TTE47/)  

## 📊 Results  

| Dataset | Evaluation | Accuracy |
|---------|------------|----------|
| **TTE47 (Test Split)** | Overall Accuracy | **94.1%** |
| **TMED2** | Fold 0 | 98.10% |
|         | Fold 1 | 98.17% |
|         | Fold 2 | 97.81% |
|         | **Average (3 folds)** | **98.03%** |

## 📂 Source & Training  

- **Dataset:** [TTE47](https://www.thrive-centre.com/datasets/TTE47/) (91,139 echocardiographic images across 47 labeled views)  
- **Labeling:** Expert cardiologist annotations (multi-expert validation for test set)  
- **Splits:** Official training (76,589), validation (9,103), and test (5,447) with no patient overlap  
- **Image Size:** Resized to **224 × 224 × 3** for model compatibility  
- **Architecture:** Pretrained **Xception** backbone with contrastive learning (LogSum loss), fine-tuned using **Cross-Entropy loss**  
- **Reproducibility:** JSON files (`class_lookup.json`, `data_split.json`) provided to align predictions with official labels and splits
- **Training:** Preshen Naidoo, Thrive Centre


---

## Import EchoView47 Contrastive Encoder 

```python
from echoforge import load_model

# Load any model by name from the registry
model = load_model("EchoView47_contrastive_encoder", pretrained=True)
```

---

---

## Import EchoView47 Classifier 

```python
from echoforge import load_model

# Load any model by name from the registry
model = load_model("EchoView47_classifier", pretrained=True)
```

---

---

## Import EchoView47 TMED2 Classifier 

```python
from echoforge import load_model

# Load any model by name from the registry
model = load_model("EchoView47_TMED2", pretrained=True)
```

---

---

## Import baseline classifiers trained on TTE47 (without contrastive pretraining) 

```python
from echoforge import load_model


# Example: Load Xception baseline model
model = load_model("Xception_TTE47_baseline", pretrained=True)

# Other available models include:
# - SwinTransformerV2_TTE47_baseline
# - EfficientNetV2_TTE47_baseline
# - ViT_TTE47_baseline
# - ConvNeXt_TTE47_baseline
# - DenseNet121_TTE47_baseline
# - ResNet50_TTE47_baseline
# - ResNet101_TTE47_baseline
```

---

## 📝 Notes  

- The model expects input images of shape **(224, 224, 3)**.  
- The `class_lookup.json` file is required to map classifier outputs to echocardiographic view names.  
- Use the official `data_split.json` to ensure reproducible training, validation, and testing.  
  



 
