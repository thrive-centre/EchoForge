import tensorflow as tf
from echoforge.utils.loader import download_model

def build_echofine_contrastive_encoder(input_shape=(224, 224, 3), pretrained=True):
    """
    Builds or loads the EchoFine contrastive encoder.

    Args:
        input_shape (tuple): Shape of the input image.
        pretrained (bool): If True, loads the model from Hugging Face.

    Returns:
        tf.keras.Model: Encoder model that outputs 2048-D feature embeddings.
    """
    if pretrained:
        print("\n Loading pretrained EchoFine contrastive encoder from Hugging Face...")
        model_path = download_model("EchoFine_contrastive_encoder")
        return tf.keras.models.load_model(model_path)

    print("\n Building EchoFine contrastive encoder architecture from scratch...")

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(1./255)(inputs)
    x = tf.keras.applications.Xception(include_top=False, weights='imagenet')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    model = tf.keras.Model(inputs, x, name="EchoFine_contrastive_encoder")
    return model
