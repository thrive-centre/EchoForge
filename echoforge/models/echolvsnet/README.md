# 🫀 EchoLVSNet - Left Ventricular Segmentation 

This project implements a robust deep learning segmentation model **EchoLVSNet** to perform **segmentation of the left ventricle.** It uses a **U-Net architecture** built in TensorFlow and supports **transfer learning** from a pre-trained model.

---
## 🔁 Transfer Learning

This project supports loading a pre-trained model developed by one of the THRIVE PhD students. The model was trained on a similar left ventricular segmentation task on the UNITY Dataset. 

---

## 🧠 Model Overview

- **Architecture:** U-Net (with encoder-decoder structure and skip connections)
- **Input Shape:** 512 x 512 grayscale PNG frames
- **Output:** Predicted segmentation mask of the left ventricle
- **Loss Function:** Binary Crossentropy + Dice Loss
- **Metrics:** Dice Coefficient, IoU & pixel-wise accuracy
- **Dataset(s):** Trained on UNITY. EchoNet-Dynamic, HMC-QU, CAMUS

---

### Input Shape
```python
inputs = tf.keras.Input(shape=(512, 512, 1))
x = tf.keras.layers.Rescaling(1./255)(inputs)
```
**What it does**
- Defines the input layer of the model.
- Accepts grayscale echocardiographic images with height and width of 512 pixels.
- The last dimension 1 indicates single-channel (grayscale) images
- Normalises pixel values from the original range of [0, 255] to [0.0, 1.0].

---

- The model expects input tensors of type *float32*

- If you’re passing raw uint8 images (e.g. PNGs), they will be automatically normalised due to the Rescaling(1./255) layer.

- Use preprocessing functions like tf.io.decode_png() with channels=1, and ensure that the resulting tensor is cast to float32.

### Convolutional Blocks
```python
def conv_block(x, filters):
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x
```

This function defines a two-layer convolutional block used throughout both the encoder and decoder stages of the network.

**Each conv_block() performs:**
  - Two 2D Convolutions with 3×3 filters and same padding.
  - Batch Normalization after each convolution to normalize intermediate outputs and improve gradient flow.
  - ReLU Activation after each normalization to introduce non-linearity and enable the model to learn complex patterns.
These blocks allow the network to extract low-level and mid-level spatial features from the input image, and are stacked to build deeper representations as the model grows.

 
### Encoder Blocks
```python
def encoder_block(x, filters):
    f = conv_block(x, filters)
    p = tf.keras.layers.MaxPooling2D((2, 2))(f)
    return f, p
```
**Actions performed:**
- 2 × Conv2D layers (3×3 kernels)
- MaxPooling for spatial downsampling
- Batch normalization and ReLU activation are not shown directly in encoder_block(), but they are used inside the conv_block() function, which encoder_block() calls

Batch Normalisation stabilises and speeds up training by normalising feature maps.
ReLU Activation introduces non-linearity and mitigates vanishing gradients.

### Bottleneck (Bridge)
```python
b = conv_block(p5, 1024)
```
The bottleneck connects encoder and decoder paths with a deeper 1024-filter representation.

### Decoder Blocks
```python
  def decoder_block(x, skip, filters):
        x = tf.keras.layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
        x = tf.keras.layers.Concatenate()([x, skip])
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        return x
```
**Actions performed:**

**1. Upsampling**
- Uses Conv2DTranspose to double the spatial resolution of the feature map.Filter count is set by the filters parameter.
- Filter count is set by the filters parameters

**2. Skip Connection**
- Concatenates the upsampled features with the corresponding encoder features (from the same depth level).
- This helps preserve spatial context and fine details lost during downsampling.

**3. Normalisation & Activation**
- Applies BatchNormalization followed by ReLU activation to the merged features.
- Ensures stable gradients and introduces non-linearity.

**4. Convolutional Refinement**
- Two Conv2D layers (3×3 kernels) each followed by BatchNorm and ReLU.
- These refine the upsampled features and help recover spatial structure lost in encoding

### Output layer 
```python
tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name='output')(d5)
```
- Produces a 1-channel binary mask
- Values range between [0, 1] (via sigmoid)

### Architecture 

| Block     | Feature Maps | Output Shape | Components            |
| --------- | ------------ | ------------ | --------------------- |
| Input     | 1            | 512×512      | -                     |
| Encoder 1 | 32           | 256×256      | Conv → BN → ReLU      |
| Encoder 2 | 64           | 128×128      | Conv → BN → ReLU      |
| Encoder 3 | 128          | 64×64        | Conv → BN → ReLU      |
| Encoder 4 | 256          | 32×32        | Conv → BN → ReLU      |
| Encoder 5 | 512          | 16×16        | Conv → BN → ReLU      |
| Bridge    | 1024         | 16×16        | Conv → BN → ReLU      |
| Decoder 1 | 512          | 32×32        | TransConv + Skip      |
| Decoder 2 | 256          | 64×64        | TransConv + Skip      |
| Decoder 3 | 128          | 128×128      | TransConv + Skip      |
| Decoder 4 | 64           | 256×256      | TransConv + Skip      |
| Decoder 5 | 32           | 512×512      | TransConv + Skip      |
| Output    | 1            | 512×512      | Conv2D (1×1, sigmoid) |


---

### 📈 Results - Test Set

- **Mean Dice Coefficient:** 0.92
- **IoU score:** 0.85

---

### 📥 How to Use

```python
from echoforge import load_model
model = load_model("EchoLVSNet", pretrained=True)
```
---

### 🔍 Model Summary

```
Total model:   31125921 parameters
Trainable params: 31109921 parameters
Non-trainable params: 16000 parameters
```

---

## 📤 Source

- Dataset: EchoNet-Dynamic - https://echonet.github.io/dynamic/
- Training by: THRIVE Research Centre

## 📎 Notes
- The model expects input images of shape (512, 512, 1), representing single-channel grayscale data typically used in echocardiographic imaging. The output is also single-channel, suitable for binary segmentation tasks.

- A custom learning rate scheduler (WarmUpCosine) is used, which is not built into the standard Keras API. Attempting to load the model with compile=True may trigger the following error:
Unknown decay: 'WarmUpCosine'.

- To avoid this, load the model with compile=False, then manually compile it using your own optimizer, learning rate, and loss configuration.
