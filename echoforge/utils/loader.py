# import os
# import json
# from huggingface_hub import hf_hub_download
# from dotenv import load_dotenv

# load_dotenv()  # Load HF_TOKEN from .env if it exists

# def download_model(model_name, save_path="~/.echoforge/models"):
#     save_path = os.path.expanduser(save_path)
#     os.makedirs(save_path, exist_ok=True)

#     with open(os.path.join(os.path.dirname(__file__), '../config/registry.json')) as f:
#         registry = json.load(f)

#     filename = registry[model_name]["filename"]
#     repo_id = registry[model_name]["repo_id"]
#     token = os.environ.get("HF_TOKEN")

#     local_path = os.path.join(save_path, filename)

#     if not os.path.exists(local_path):
#         print(f" Downloading {filename} from Hugging Face (private repo)...")
#         downloaded_path = hf_hub_download(
#             repo_id=repo_id,
#             filename=filename,
#             cache_dir=save_path,
#             token=token
#         )
#         os.rename(downloaded_path, local_path)

#     return local_path


import os
import json
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

load_dotenv()

def download_model(model_name, save_path="~/.echoforge/models"):
    save_path = os.path.expanduser(save_path)
    os.makedirs(save_path, exist_ok=True)

    # Load registry
    with open(os.path.join(os.path.dirname(__file__), '../config/registry.json')) as f:
        registry = json.load(f)

    # Get model details
    filename = registry[model_name]["filename"]
    repo_id = registry[model_name]["repo_id"]
    token = os.environ.get("HF_TOKEN")

    # Check if model already exists locally
    local_path = os.path.join(save_path, filename)
    if os.path.exists(local_path):
        print(" Model already cached locally.")
        return local_path

    # Otherwise, download it
    print(" Downloading model from Hugging Face...")
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=save_path,
        token=token
    )

    # Move from cache to final location
    os.rename(downloaded_path, local_path)

    return local_path
