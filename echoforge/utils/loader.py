import os
import json
import shutil

from huggingface_hub import hf_hub_download, snapshot_download

_REGISTRY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "config", "registry.json"
)


def _normalize(name):
    return name.lower().replace("-", "_")


def load_registry():
    """Return the registry as {model_name: {"repo_id": ..., "filename": ...}}."""
    with open(_REGISTRY_PATH) as f:
        return json.load(f)


def resolve_name(model_name, registry=None):
    """Map a user supplied name (case / '-' vs '_' insensitive) to a registry key."""
    registry = registry if registry is not None else load_registry()
    if model_name in registry:
        return model_name
    wanted = _normalize(model_name)
    for key in registry:
        if _normalize(key) == wanted:
            return key
    # backwards compatibility with the old EchoView47 naming
    legacy = wanted.replace("echoview47", "echofine")
    for key in registry:
        if _normalize(key) == legacy:
            return key
    raise ValueError(
        f"Model '{model_name}' not found in registry. Available: {sorted(registry)}"
    )


def _cache_dir():
    path = os.path.expanduser(os.environ.get("ECHOFORGE_MODEL_DIR", "~/.echoforge/models"))
    os.makedirs(path, exist_ok=True)
    return path


def download_model(model_name):
    """Download a model from Hugging Face (cached) and return its local path.

    Registry entries use ``repo_id`` plus either:
      * ``filename``  - a single file (.h5 / .keras / .weights.h5), or
      * no ``filename`` (null/empty) - the whole repo, e.g. a SavedModel directory.
    An optional ``subfolder`` selects a directory inside the repo.
    """
    registry = load_registry()
    key = resolve_name(model_name, registry)
    info = registry[key]
    repo_id = info["repo_id"]
    filename = info.get("filename")
    subfolder = info.get("subfolder")
    token = os.environ.get("HF_TOKEN")

    if "PLACEHOLDER" in repo_id or "PLACEHOLDER" in (filename or ""):
        raise ValueError(
            f"Model '{key}' is not published yet. Update its repo_id/filename in "
            "echoforge/config/registry.json."
        )

    cache = _cache_dir()

    if filename:
        local_path = os.path.join(cache, repo_id.replace("/", "__"), subfolder or "", filename)
        if os.path.exists(local_path):
            print(" Model already cached locally:", local_path)
            return local_path
        print("⬇  Downloading model from Hugging Face...")
        downloaded = hf_hub_download(
            repo_id=repo_id, filename=filename, subfolder=subfolder, token=token
        )
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        shutil.copy(downloaded, local_path)
        print(f" Saving model to: {local_path}")
        return local_path

    print("⬇  Downloading model repository from Hugging Face...")
    patterns = [f"{subfolder}/*"] if subfolder else None
    root = snapshot_download(repo_id=repo_id, token=token, allow_patterns=patterns)
    return os.path.join(root, subfolder) if subfolder else root


def load_full_model(model_name, pretrained=True, compile=False, custom_objects=None):
    """Download and load the COMPLETE Keras model (architecture + weights).

    With ``pretrained=False`` the same architecture is returned with freshly
    initialised weights, ready for retraining.
    """
    import tensorflow as tf

    path = download_model(model_name)
    if os.path.isdir(path) and not os.path.exists(os.path.join(path, "saved_model.pb")):
        for ext in (".keras", ".h5"):
            found = sorted(f for f in os.listdir(path) if f.endswith(ext))
            if found:
                path = os.path.join(path, found[0])
                break
    model = tf.keras.models.load_model(path, compile=compile, custom_objects=custom_objects)
    if not pretrained:
        fresh = tf.keras.models.clone_model(model)
        return fresh
    return model
