import os
import tensorflow as tf
from echoforge.utils.loader import download_model

def build_echo2d_model(input_shape=(224, 224, 3), num_classes=47, pretrained=True):
    """
    Builds or loads the Echo2DClassifier model.

    Args:
        input_shape (tuple): Image input size
        num_classes (int): Number of output classes
        pretrained (bool): Whether to load pretrained model from Hugging Face

    Returns:
        tf.keras.Model
    """
    if pretrained:
        print(" Loading pretrained model from Hugging Face...")
        model_path = download_model("Echo2DClassifier")
        return tf.keras.models.load_model(model_path)

    print(" Building Echo2DClassifier architecture without pretrained weights...")
    base = tf.keras.applications.Xception(
        include_top=False,
        weights='imagenet',  # You can also make this optional
        input_shape=input_shape
    )
    model = tf.keras.Sequential([
        base,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model
