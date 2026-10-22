
from .segmentation.models.echolvsnet.echolvsnet import build_echolvsnet
from .classification.models.EchoFine.EchoFine_contrastive_encoder import build_echofine_contrastive_encoder
from .classification.models.EchoFine.EchoFine_classifier import build_echofine_classifier
from .classification.models.EchoFine.EchoFine_TMED2 import build_echofine_tmed2
from .classification.models.EchoSDNet.EchoSDNet import build_EchoSDNet

def list_models():
    """Return the names of all models in the registry."""
    from .utils.loader import load_registry
    return sorted(load_registry())


def load_model(name, pretrained=True):
    """Load a complete model (architecture + weights) by registry name.

    Built-in models keep their dedicated builders; every other registry entry is
    loaded as a full Keras model straight from its Hugging Face location.
    """
    builders = {
        "echofine_contrastive_encoder": build_echofine_contrastive_encoder,
        "echofine_classifier": build_echofine_classifier,
        "echofine_tmed2": build_echofine_tmed2,
        "echosdnet": build_EchoSDNet,
        "echolvsnet": build_echolvsnet,
    }
    key = name.lower().replace("echoview47", "echofine")
    builder = builders.get(key)
    if builder is not None:
        return builder(pretrained=pretrained)

    from .utils.loader import load_full_model
    return load_full_model(name, pretrained=pretrained)
