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

## ✅ Classification Performance

- **Dataset**: UNITY-Classification-B (47-class high-granularity labeling)
- **Test Accuracy**: **94.2%** on the held-out test set

---

## ✅ Classes Covered

Echo2DClassifier is trained to predict **47 unique views** from echocardiographic scans.

👉 For full class breakdown and examples, see the full dataset description:  
📄 [Download Word file](./dataset_description/Classification%20B%20Dataset%20-%20Detailed%20Description.docx)

---

## 📥 How to Use

```python
from echoforge import load_model
model = load_model("Echo2DClassifier", pretrained=True)
```

Or load only the architecture:

```python
from echoforge.models import Echo2DClassifier
model = Echo2DClassifier(pretrained=False)
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
