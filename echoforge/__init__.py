
from .segmentation.models.echolvsnet.echolvsnet import build_echolvsnet
from .classification.models.EchoView47.EchoView47_contrastive_encoder import build_echoview47_contrastive_encoder
from .classification.models.EchoView47.EchoView47_classifier import build_echoview47_classifier
from .classification.models.EchoView47.EchoView47_TMED2 import build_echoview47_tmed2
from .classification.models.EchoSDNet import build_EchoSDNet

def load_model(name, pretrained=True):
    if name == "EchoView47_contrastive_encoder":
        return build_echoview47_contrastive_encoder(pretrained=pretrained)
    if name == "EchoView47_classifier":
        return build_echoview47_classifier(pretrained=pretrained)
    if name == "EchoView47_TMED2":
        return build_echoview47_tmed2(pretrained=pretrained)
    if name == "EchoSDNet":
        return build_EchoSDNet(pretrained=pretrained)
    if name.lower() == "echolvsnet":
        return build_echolvsnet(pretrained=pretrained)
    else:
        raise ValueError(f"Model '{name}' not found.")

