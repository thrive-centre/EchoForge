
from .segmentation.models.echolvsnet.echolvsnet import build_echolvsnet
from .classification.models.EchoFine.EchoFine_contrastive_encoder import build_echofine_contrastive_encoder
from .classification.models.EchoFine.EchoFine_classifier import build_echofine_classifier
from .classification.models.EchoFine.EchoFine_TMED2 import build_echofine_tmed2
from .classification.models.EchoSDNet.EchoSDNet import build_EchoSDNet

def load_model(name, pretrained=True):
    builders = {
        "echofine_contrastive_encoder": build_echofine_contrastive_encoder,
        "echofine_classifier": build_echofine_classifier,
        "echofine_tmed2": build_echofine_tmed2,
        "echosdnet": build_EchoSDNet,
        "echolvsnet": build_echolvsnet,
    }
    aliases = {k.replace("echofine", "echoview47"): v for k, v in builders.items() if k.startswith("echofine")}
    builder = builders.get(name.lower()) or aliases.get(name.lower())
    if builder is None:
        raise ValueError(f"Model '{name}' not found. Available: {sorted(builders)}")
    return builder(pretrained=pretrained)
