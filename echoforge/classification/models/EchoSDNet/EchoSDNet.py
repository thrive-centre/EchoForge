import tensorflow as tf
from echoforge.utils.loader import download_model

def build_EchoSDNet(input_shape=(224, 224, 3), num_classes=47, pretrained=True):
    """
    Builds or loads the EchoView47 47-class classifier.

    Args:
        input_shape (tuple): Shape of the input image.
        num_classes (int): Number of target output classes.
        pretrained (bool): If True, loads the model from Hugging Face.

    Returns:
        tf.keras.Model: A compiled Keras model.
    """
    if pretrained:
        print("\n Loading pretrained EchoSDNet classifier from Hugging Face...")
        model_path = download_model("EchoSDNet")
        return tf.keras.models.load_model(model_path)

    print("\n Building EchoView47 classifier architecture from scratch...")

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(1./255)(inputs)
    base = tf.keras.applications.Xception(include_top=False, weights='imagenet')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(base)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs, name="EchoView47_classifier")
    return model
