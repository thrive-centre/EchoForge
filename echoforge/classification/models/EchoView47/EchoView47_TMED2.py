import tensorflow as tf
from echoforge.utils.loader import download_model
from echoforge.classification.models.EchoView47.EchoView47_contrastive_encoder import build_echoview47_contrastive_encoder

def build_echoview47_tmed2(input_shape=(224, 224, 3), num_classes=5, pretrained=True):
    """
    Builds or loads the EchoView47-TMED2 classifier.

    This model is fine-tuned on the TMED2 dataset using a pretrained contrastive encoder.

    Args:
        input_shape (tuple): Shape of the input image.
        num_classes (int): Number of output classes (default is 5 for TMED2).
        pretrained (bool): If True, loads the model from Hugging Face.

    Returns:
        tf.keras.Model: A Keras model for TMED2 classification.
    """
    if pretrained:
        print("\n Loading pretrained EchoView47-TMED2 model from Hugging Face...")
        model_path = download_model("EchoView47_TMED2")
        return tf.keras.models.load_model(model_path)

    print("\n Building EchoView47-TMED2 classifier architecture from scratch...")

    base_encoder = build_echoview47_contrastive_encoder(input_shape=input_shape, pretrained=False)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(base_encoder.output)

    model = tf.keras.Model(inputs=base_encoder.input, outputs=outputs, name="EchoView47_TMED2")
    return model
