from echoforge.utils.loader import load_full_model


def build_echopdnet(pretrained=True):
    """Load the complete EchoPDNet model from the configured registry."""
    return load_full_model("EchoPDNet", pretrained=pretrained)


__all__ = ["build_echopdnet"]
