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
  - Only the final block (conv5) unfrozen for fine-tuning  
  - Residual blocks with skip connections mitigate vanishing gradients  

**Custom Classification Head**  
  - Global Average Pooling → 2048-dim feature map  
  - Dense layer (128 units, ReLU)  
  - Dense layer (64 units, ReLU)  
  - Dropout (0.5) for regularisation  
  - Output Dense layer (2 units, Softmax)  


