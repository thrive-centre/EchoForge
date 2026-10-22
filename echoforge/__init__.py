
from .segmentation.models.echolvsnet.echolvsnet import build_echolvsnet
from .classification.models.EchoView47.EchoView47_contrastive_encoder import build_echoview47_contrastive_encoder
from .classification.models.EchoView47.EchoView47_classifier import build_echoview47_classifier
from .classification.models.EchoView47.EchoView47_TMED2 import build_echoview47_tmed2
from .classification.models.EchoSDNet.EchoSDNet import build_EchoSDNet

def load_model(name, pretrained=True):
    builders = {
        "echoview47_contrastive_encoder": build_echoview47_contrastive_encoder,
        "echoview47_classifier": build_echoview47_classifier,
        "echoview47_tmed2": build_echoview47_tmed2,
        "echosdnet": build_EchoSDNet,
        "echolvsnet": build_echolvsnet,
    }
    builder = builders.get(name.lower())
    if builder is None:
        raise ValueError(f"Model '{name}' not found. Available: {sorted(builders)}")
    return builder(pretrained=pretrained)
