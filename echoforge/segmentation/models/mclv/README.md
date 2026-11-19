# MCLV - Multi-Center Left Ventricle Segmentation

MCLV is a TensorFlow/Keras model for segmenting the left ventricle in grayscale echocardiographic images.

---

## Model Overview

- **Architecture:** EchoLVSNet-based U-Net encoder-decoder with skip connections
- **Task:** Binary segmentation of the left ventricle
- **Input Shape:** `(batch_size, 256, 256, 1)`
- **Output Shape:** `(batch_size, 256, 256, 1)`
- **Output:** Pixel-wise probability map for the left ventricle
- **Framework:** TensorFlow/Keras
- **Parameters:** 31,125,921

The architecture uses the EchoLVSNet U-Net design with convolution, batch normalisation, ReLU activation, max pooling, skip connections, and transposed convolutions.

---

## Model Input

```python
inputs = tf.keras.Input(shape=(256, 256, 1))
x = tf.keras.layers.Rescaling(1./255)(inputs)
```

- Accepts single-channel grayscale echocardiographic frames.
- Expects images resized to `256 x 256`.
- Accepts float32 tensors in the range `[0, 1]`.
- When loading 8-bit images, divide pixel values by `255`.

---

## Preprocessing

```python
import cv2
import numpy as np

frame = cv2.imread("cardiac_frame.png", cv2.IMREAD_GRAYSCALE)
frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR)
frame = frame.astype(np.float32) / 255.0

input_array = np.expand_dims(frame, axis=(0, -1))
```

The resulting input shape is `(1, 256, 256, 1)`.

---

## Architecture

| Block | Feature Maps | Output Shape | Components |
|---|---:|---|---|
| Input | 1 | 256 x 256 | Rescaling |
| Encoder 1 | 32 | 128 x 128 | Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> MaxPool |
| Encoder 2 | 64 | 64 x 64 | Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> MaxPool |
| Encoder 3 | 128 | 32 x 32 | Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> MaxPool |
| Encoder 4 | 256 | 16 x 16 | Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> MaxPool |
| Encoder 5 | 512 | 8 x 8 | Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> MaxPool |
| Bottleneck | 1024 | 8 x 8 | Conv -> BN -> ReLU -> Conv -> BN -> ReLU |
| Decoder 1 | 512 | 16 x 16 | Transposed Conv -> Skip -> Conv -> Conv |
| Decoder 2 | 256 | 32 x 32 | Transposed Conv -> Skip -> Conv -> Conv |
| Decoder 3 | 128 | 64 x 64 | Transposed Conv -> Skip -> Conv -> Conv |
| Decoder 4 | 64 | 128 x 128 | Transposed Conv -> Skip -> Conv -> Conv |
| Decoder 5 | 32 | 256 x 256 | Transposed Conv -> Skip -> Conv -> Conv |
| Output | 1 | 256 x 256 | Conv2D 1 x 1 -> Sigmoid |

---

## Training Information

- **Training scope:** Multi-center cardiac ultrasound datasets
- **Checkpoint selection:** Validation-selected best checkpoint
- **Training data:** The supplied documentation does not specify named datasets or dataset splits.

No dataset names or split details are added without supporting model documentation.

---

## Results

The following results are MCLV benchmark scores.

| Dataset | Images | Dice Mean | Dice Std | IoU Mean | IoU Std |
|---|---:|---:|---:|---:|---:|
| Multix | 200 | 0.934 | 0.026 | 0.877 | 0.045 |
| Unity | 375 | 0.925 | 0.041 | 0.862 | 0.064 |
| EchoNet | 3,062 | 0.913 | 0.050 | 0.844 | 0.069 |
| CAMUS | 132 | 0.942 | 0.032 | 0.893 | 0.054 |
| HMC | 263 | 0.957 | 0.020 | 0.919 | 0.034 |

---

## How to Use

```python
from echoforge import load_model

model = load_model("MCLV", pretrained=True)
probability_map = model.predict(input_array, verbose=0)
binary_mask = (probability_map[0, :, :, 0] > 0.5).astype("uint8")
```

The model loads `mclv_tf213.h5` from the configured Hugging Face repository.

### Resize the Mask

```python
mask = cv2.resize(
    binary_mask,
    (original_width, original_height),
    interpolation=cv2.INTER_NEAREST,
)
```

---

## Complete Example

```python
import cv2
import numpy as np
from echoforge import load_model

frame = cv2.imread("cardiac_frame.png", cv2.IMREAD_GRAYSCALE)
original_height, original_width = frame.shape[:2]

frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR)
frame = frame.astype(np.float32) / 255.0
input_array = np.expand_dims(frame, axis=(0, -1))

model = load_model("MCLV", pretrained=True)
probability_map = model.predict(input_array, verbose=0)

binary_mask = (probability_map[0, :, :, 0] > 0.5).astype(np.uint8)
binary_mask = cv2.resize(
    binary_mask,
    (original_width, original_height),
    interpolation=cv2.INTER_NEAREST,
)

cv2.imwrite("lv_segmentation_mask.png", binary_mask * 255)
```

---

## Model Details

- **Base architecture:** EchoLVSNet
- **Input channels:** 1
- **Output channels:** 1
- **Loss Function:** Dice Loss + Cross-Entropy
- **Optimizer:** Adam
- **Batch Size:** 32
- **Checkpoint selection:** Validation-selected best checkpoint
- **Model file:** `mclv_tf213.h5`

---

## Notes

- The model expects float32 input and produces float32 pixel-wise probabilities.
- Use a threshold of `0.5` to convert probabilities into a binary mask.
- Resize output masks back to the original frame dimensions with nearest-neighbour interpolation.
- Performance can vary with ultrasound equipment, image quality, and acquisition view.
