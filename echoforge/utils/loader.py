import os
import json
import shutil
from huggingface_hub import hf_hub_download

def download_model(model_name):
    # ✅ Use env var if set, else default
    save_path = os.environ.get("ECHOFORGE_MODEL_DIR", "~/.echoforge/models")
    save_path = os.path.expanduser(save_path)  # expand ~ if needed
    os.makedirs(save_path, exist_ok=True)

    # Load registry.json
    registry_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "registry.json")
    with open(registry_path) as f:
        registry = json.load(f)

    # Get model info
    if model_name not in registry:
        raise ValueError(f"Model '{model_name}' not found in registry. Available: {sorted(registry)}")
    filename = registry[model_name]["filename"]
    repo_id = registry[model_name]["repo_id"]
    token = os.environ.get("HF_TOKEN")

    if "PLACEHOLDER" in repo_id or "PLACEHOLDER" in filename:
        raise ValueError(
            f"Model '{model_name}' is not published yet. Update its repo_id/filename in echoforge/config/registry.json."
        )

    # Final file path
    local_path = os.path.join(save_path, filename)

    # ✅ If already downloaded, return
    if os.path.exists(local_path):
        print(" Model already cached locally:", local_path)
        return local_path

    # ⬇️ Otherwise download
    print("⬇  Downloading model from Hugging Face...")
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=token
    )

    # ✅ Copy to ECHOFORGE_MODEL_DIR
    print(f" Saving model to: {local_path}")
    shutil.copy(downloaded_path, local_path)

    return local_path