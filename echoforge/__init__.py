from .models.echo2dclassifier.echo2dclassifier import build_echo2d_model
from .models.echolvsnet.echolvsnet import build_echolvsnet

def load_model(name, pretrained=True):
    if name.lower() == "echo2dclassifier":
        return build_echo2d_model(pretrained=pretrained)
    elif name.lower() == "echolvsnet":
        return build_echolvsnet(pretrained=pretrained)
    else:
        raise ValueError(f"Model '{name}' not found.")

