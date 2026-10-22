"""Add every published registry model to a Hugging Face collection.

    python tools/add_to_collection.py thrive-2025/echoforge-<id>          # dry run
    python tools/add_to_collection.py thrive-2025/echoforge-<id> --execute

The slug is the end of the collection URL: huggingface.co/collections/<slug>
"""
import argparse
import json
import os

REGISTRY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "echoforge", "config", "registry.json")

ap = argparse.ArgumentParser()
ap.add_argument("slug", help="collection slug, e.g. thrive-2025/echoforge-65f0...")
ap.add_argument("--execute", action="store_true")
args = ap.parse_args()

with open(REGISTRY) as f:
    registry = json.load(f)
repos = sorted({v["repo_id"] for v in registry.values() if "PLACEHOLDER" not in v["repo_id"]})
print(f"{len(repos)} repos")

if not args.execute:
    print("\n".join(repos) + "\n\nDry run only. Add --execute to add them.")
else:
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    for r in repos:
        api.add_collection_item(args.slug, item_id=r, item_type="model", exists_ok=True)
        print("added", r)
