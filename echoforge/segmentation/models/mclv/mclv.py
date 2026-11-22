import json

import h5py
import tensorflow as tf

from echoforge.utils.loader import download_model


def _keras_tensor_inputs(value):
    if isinstance(value, list):
        return [item for child in value for item in _keras_tensor_inputs(child)]
    if isinstance(value, dict) and value.get("class_name") == "__keras_tensor__":
        return [value["config"]["keras_history"] + [{}]]
    raise ValueError("Unsupported MCLV Keras 3 model input configuration")


def _load_mclv(model_path):
    with h5py.File(model_path, "r") as model_file:
        version = model_file.attrs.get("keras_version", "")
        if isinstance(version, bytes):
            version = version.decode("utf-8")
        if not str(version).startswith("3."):
            return tf.keras.models.load_model(model_path, compile=False)
        config = json.loads(model_file.attrs["model_config"])

    # Keras 3 HDF5 graph nodes are not understood by the supported Keras 2.13.
    for layer in config["config"]["layers"]:
        if layer["class_name"] == "InputLayer":
            layer["config"]["batch_input_shape"] = layer["config"].pop("batch_shape")
        nodes = []
        for node in layer["inbound_nodes"]:
            kwargs = node["kwargs"]
            if any(key != "mask" or value is not None for key, value in kwargs.items()):
                raise ValueError("Unsupported MCLV Keras 3 model node arguments")
            nodes.append(_keras_tensor_inputs(node["args"]))
        layer["inbound_nodes"] = nodes
    model = tf.keras.models.model_from_json(json.dumps(config))
    model.load_weights(model_path)
    return model


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
    model = _load_mclv(model_path)

    if pretrained:
        return model

    return tf.keras.models.clone_model(model)
