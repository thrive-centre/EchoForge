from .models.echo2dclassifier.echo2dclassifier import build_echo2d_model

def load_model(name, pretrained=True):
    if name.lower() == "echo2dclassifier":
        return build_echo2d_model(pretrained=pretrained)
    else:
        raise ValueError(f"Model '{name}' not found.")
