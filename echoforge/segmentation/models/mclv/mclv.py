import tensorflow as tf

from echoforge.utils.loader import download_model


def build_mclv(input_shape=(256, 256, 1), pretrained=True):
    """
    Load or initialise MCLV, a multi-center left ventricular segmentation model.

    Args:
        input_shape (tuple): Input image shape.
        pretrained (bool): Whether to load pretrained weights from Hugging Face.

    Returns:
        tf.keras.Model: MCLV left ventricular segmentation model.
    """
    model_path = download_model("MCLV")
    model = tf.keras.models.load_model(model_path, compile=False)

    if pretrained:
        return model

    return tf.keras.models.clone_model(model)
