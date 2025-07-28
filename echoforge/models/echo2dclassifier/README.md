# Echo2DClassifier

**Echo2DClassifier** is the first official model in the EchoForge library. It is a robust deep learning model trained on the UNITY-Classification-B dataset to perform multi-class 2D echocardiographic view classification.

---

## 📌 Model Details

- **Architecture**: Xception (base) + Flatten + Dense(47)
- **Input Size**: 224 x 224 x 3
- **Output**: One of 47 standard echocardiographic views
- **Training**: Trained on high-quality labels from UNITY-Classification-B
- **Saved Format**: Full `.h5` model (architecture + weights)
- **Expected Input Format**: Images must be resized to `224x224x3` before passing to the model. This can be done using a `tf.data` pipeline with `dataset.map()` and `tf.image.resize`.

---


```python
   base = tf.keras.applications.Xception(
        include_top=False,
        weights='imagenet',  # You can also make this optional
        input_shape=input_shape
    )
```

| Parameter                        | Description                                                                                                                                                                                                                                                                    |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `tf.keras.applications.Xception` | Loads the **Xception** architecture — a deep convolutional neural network with depthwise separable convolutions, known for high performance on image tasks.                                                                                                                    |
| `include_top=False`              | This **excludes** the final classification layers (fully connected layers). You're only using the **convolutional base**, which outputs feature maps instead of class predictions. Useful when you want to add your own custom head for tasks like segmentation or regression. |
| `weights='imagenet'`             | Loads **pre-trained weights from ImageNet**, a large-scale dataset. This allows your model to start with learned filters instead of training from scratch, speeding up convergence and improving accuracy.                                                                     |
| `input_shape=input_shape`        | Specifies the shape of your input images (e.g., `(224, 224, 3)`). It must match the expected size of the Xception base.                                                                                                                                                        |

```python
  model = tf.keras.Sequential([
        base,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model
```
### Functional Overview 
1. Uses a pre-trained feature extractor (base, e.g., Xception without the top layers)
2. Flattens the extracted features
3. Classifies the input into one of the num_classes using a Dense softmax layer

| Parameter                                                  | Description                                                                                                                     |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `tf.keras.Sequential([...])`                               | Initialises a sequential model — a linear stack of layers.                                                                      |
| `base`                                                     | A pre-trained model (like Xception) used as a feature extractor. This is the "backbone" of the model.                           |
| `tf.keras.layers.Flatten()`                                | Converts the multi-dimensional output from the CNN base into a 1D vector, preparing it for the dense classification layer.      |
| `tf.keras.layers.Dense(num_classes, activation='softmax')` | The final classification layer. It has `num_classes` neurons and uses the **softmax** activation to output class probabilities. |
| `return model`                                             | Returns the compiled model so it can be trained or evaluated.                                                                   |



## ✅ Classification Performance

- **Dataset**: UNITY-Classification-B (47-class high-granularity labeling)
- **Test Accuracy**: **94.2%** on the held-out test set

---

## ✅ Classes Covered

Echo2DClassifier is trained to predict **47 unique views** from echocardiographic scans.

👉 For full class breakdown and examples, see the full dataset description:  
📄 [Download Word file](./dataset_description/Classification%20B%20Dataset%20-%20Detailed%20Description.docx)

---

### Echo2DClassifier Architecture 

| Block                 | Output Shape             | Components                                          |
| --------------------- | ------------------------ | --------------------------------------------------- |
| **Input**             | (224 × 224 × 3)          | -                                                   |
| **Feature Extractor** | Varies based on backbone | Xception (ImageNet pretrained, `include_top=False`) |
| **Flatten**           | 1D vector                | `Flatten()`                                         |
| **Classifier**        | `(num_classes,)`         | `Dense(num_classes, activation='softmax')`          |
| **Output**            | `(47,)` (default)        | Softmax probability over class labels               |


## 📥 How to Use

```python
from echoforge import load_model
model = load_model("Echo2DClassifier", pretrained=True)
```

---

## 📤 Source & Training

- **Dataset**: UNITY-Classification-B (47-class view-labeled echocardiographic dataset)
- **Labeling**: Expert clinician-labeled 
- **Training**: Preshen, IntSaV Research Group

---

## 📎 Notes

- Model expects input shape of **(224, 224, 3)**.
- You must resize images externally using `tf.image.resize()` or in your `tf.data` pipeline.
- No resizing is performed inside the model itself.
