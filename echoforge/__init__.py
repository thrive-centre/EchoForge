
from .segmentation.models.echolvsnet.echolvsnet import build_echolvsnet
from .classification.models.EchoFine.EchoFine_contrastive_encoder import build_echofine_contrastive_encoder
from .classification.models.EchoFine.EchoFine_classifier import build_echofine_classifier
from .classification.models.EchoFine.EchoFine_TMED2 import build_echofine_tmed2
from .classification.models.EchoSDNet.EchoSDNet import build_EchoSDNet

def list_models():
    """Return the names of all models in the registry."""
    import json, os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "registry.json")
    with open(path) as f:
        return sorted(json.load(f))


def load_model(name, pretrained=True):
    builders = {
        "echofine_contrastive_encoder": build_echofine_contrastive_encoder,
        "echofine_classifier": build_echofine_classifier,
        "echofine_tmed2": build_echofine_tmed2,
        "echosdnet": build_EchoSDNet,
        "echolvsnet": build_echolvsnet,
    }
    aliases = {k.replace("echofine", "echoview47"): v for k, v in builders.items() if k.startswith("echofine")}
    key = name.lower()
    builder = builders.get(key) or aliases.get(key)
    if builder is not None:
        return builder(pretrained=pretrained)

    # Generic path: any other model listed in the registry loads straight from Hugging Face
    registry = {m.lower(): m for m in list_models()}
    if key in registry:
        if not pretrained:
            raise ValueError(f"Model '{name}' has no architecture builder; use pretrained=True.")
        import tensorflow as tf
        from .utils.loader import download_model
        return tf.keras.models.load_model(download_model(registry[key]), compile=False)

    raise ValueError(f"Model '{name}' not found. Available: {list_models()}")
