# EchoSDNet 

This project implements a deep learning classification model **EchoSDNet** to perform binary classification of echocardiographic frames into **Single Heart** and **Dual Heart views.** The model is built on a ResNet50 backbone with custom classification layers and leverages transfer learning from ImageNet.

The purpose of this model is to provide a robust upstream classification step for downstream echocardiographic tasks, particularly image quality assessment, where accurate distinction between single and dual views ensures reliable inputs for automated diagnostic pipelines. 

---

## Dataset

### **UNITY**

The dataset used for this study was obtained through collaboration with THRIVE Research Centre. UNITY represents one of the UK’s largest repositories of clinically annotated echocardiographic frames.

- **Initial subset**: 733 images (706 single-view, 27 dual-view)  
- **Expanded dataset**: 1,832 images spanning 59 view categories  
- **Final dataset (post-balancing & verification)**:  
  - 1,275 images total  
  - 706 single-view  
  - 569 dual-view  
- **Split strategy**: 70% training, 15% validation, 15% testing (stratified to preserve class proportions)

All images were anonymised, ethically approved, and preprocessed to **224×224 RGB resolution**, with pixel values normalised to [0, 1].

---

## Model Architecture

EchoSDNet employs a **ResNet50 backbone** (pre-trained on ImageNet) with a custom classification head:

**ResNet50 Backbone**  
  - Functions as a feature extractor  
  - Only the **final block (conv5)** unfrozen for fine-tuning  
  - Residual blocks with skip connections mitigate vanishing gradients  

**Custom Classification Head**  
  - Global Average Pooling → 2048-dim feature map  
  - Dense layer (128 units, ReLU)  
  - Dense layer (64 units, ReLU)  
  - Dropout (0.5) for regularisation  
  - Output Dense layer (2 units, Softmax)
    
| Block / Layer Group                 | Key Layers (examples)                               | Output Shape        | Notes |
|------------------------------------|------------------------------------------------------|---------------------|------|
| **Input**                          | `input_1`                                            | (None, 224, 224, 3) | RGB images (resized externally). |
| **Stem**                           | `ZeroPadding2D → Conv2D(64,7×7,stride=2) → BN → ReLU → MaxPool` | (None, 56, 56, 64)  | Standard ResNet50 stem. |
| **Stage conv2_x**                  | 3 × bottleneck blocks (`conv2_block1..3`)            | (None, 56, 56, 256) | First residual stage. |
| **Stage conv3_x**                  | 4 × bottleneck blocks (`conv3_block1..4`)            | (None, 28, 28, 512) | Downsamples spatially. |
| **Stage conv4_x**                  | 6 × bottleneck blocks (`conv4_block1..6`)            | (None, 14, 14, 1024)| Deep feature extraction. |
| **Stage conv5_x** *(trainable)*    | 3 × bottleneck blocks (`conv5_block1..3`)            | (None, 7, 7, 2048)  | Final ResNet stage; unfrozen for fine-tuning. |
| **Pooling**                        | `GlobalAveragePooling2D`                             | (None, 2048)        | Converts feature maps → vector. |
| **Head (Dense → Dense → Dropout)** | `Dense(128, ReLU) → Dense(64, ReLU) → Dropout(0.5)`  | (None, 64)          | Task-specific classifier head. |
| **Output**                         | `Dense(2, Softmax)`                                  | (None, 2)           | Classes: **Single** vs **Dual**. |

**Parameters**  
- **Total:** 23,858,370 (≈ 91.01 MB)  
- **Trainable:** 15,246,658 (≈ 58.16 MB)  
- **Non-trainable:** 8,611,712 (≈ 32.85 MB)

---

### Layer Breakdown 
**Input (224 x 224 x 3)**
- Expects RGB frames pre-resized and scaled (e.g., to [0,1]). No internal resizing.

**Standard ResNet50 Stem**
- A padded 7×7 Conv→BN→ReLU followed by MaxPool quickly reduces spatial size and builds low-level edges/textures.

**Residual Stages (conv2_x → conv5_x)**

Built from **bottleneck residual blocks** (1×1 → 3×3 → 1×1 with identity/shortcut adds).
- **conv2_x (56×56×256):** preserves resolution; stabilises early features.
- **conv3_x (28×28×512):** first downsampling; captures mid-level patterns.
- **conv4_x (14×14×1024):** deepest stack; rich semantic features.
- **conv5_x (7×7×2048):** unfrozen for domain fine-tuning; adapts ImageNet features to echo imagery.
  
**GlobalAveragePooling2D → 2048-D vector**
- Spatially averages each feature map, yielding a compact, translation-robust embedding.

**Dense(128, ReLU) → Dense(64, ReLU)**
- Learns task-specific combinations of features; adds non-linearity beyond the backbone.

**Dropout(0.5)**
- Regularises the head by randomly deactivating units during training to reduce overfitting.

**Dense (2, Softmax)**
- Outputs calibrated probabilities for Single vs Dual views.

--- 

## Training Details 
- **Optimizer**: Adam (learning rate = 1e-5)
- **Loss Function**: Categorical Cross-Entropy
- **Epochs**: 20 with early stopping (patience = 5)

| Augmentation Technique              | Description                                                                  |
|-------------------------------------|------------------------------------------------------------------------------|
| *Rotation – within a range of 15°*  | Randomly rotates the image clockwise or counterclockwise by up to 15°        |
| *Width Shift – up to 10%*           | Translates the image horizontally by up to 10%                               |
| *Height Shift – up to 10%*          | Translates the image vertically by up to 10% of its height                   |
| *Zoom – up to 20%*                  | Randomly scales the image in or out by up to 20%                             |

## Results

| Metric          | Single-View | Dual-View | Overall |
|-----------------|-------------|-----------|---------|
| Precision       | 1.00        | 0.99      | 0.99    |
| Recall          | 0.99        | 1.00      | 0.99    |
| F1-Score        | 1.00        | 0.99      | 0.99    |
| Accuracy        | -           | -         | **99.48%** |
| Test Loss       | -           | -         | 0.0155  |
| ROC-AUC         | -           | -         | 1.00    |
| PR-AUC          | -           | -         | 1.00    |

**Confusion Matrix**  
- 192 total test samples  
- 191 correctly classified, 1 misclassified  

---

## Import EchoSDNet  

```python
from echoforge import load_model

# Load any model by name from the registry
model = load_model("EchoSDNet", pretrained=True)
```

---

## Notes

- EchoSDNet demonstrates **state-of-the-art performance** for binary classification of echocardiographic views.  
- Importantly, **no dual-view frames were misclassified**, reducing clinical risk in downstream workflows.  
- The model was evaluated **only** on UNITY data. 
- By providing a reliable distinction between single and dual views, EchoSDNet supports **standardisation, reproducibility, and clinical integration** of automated echo analysis pipelines.  


