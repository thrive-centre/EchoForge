"""Upload model files to Hugging Face, one repo per model, and update the registry.

Expected local layout: a root folder holding model files (.h5 / .keras), at any depth.
Each file becomes one repo named after the file (no extension), and one registry key
with the same name.

Usage (dry run by default - nothing is uploaded or written):
    python tools/upload_models.py "C:/path/to/final_models"
    python tools/upload_models.py "C:/path/to/final_models" --execute
    python tools/upload_models.py "C:/path/to/final_models" --execute --private --only ViT-S

Needs `huggingface-cli login` or the HF_TOKEN environment variable (write access).
"""
import argparse
import json
import os
import re
import sys

ORG = "thrive-2025"
EXTS = (".h5", ".keras")
REGISTRY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "echoforge", "config", "registry.json")


def repo_name(stem):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_.-")


def find_models(root):
    for d, _, files in os.walk(root):
        for f in sorted(files):
            if f.lower().endswith(EXTS):
                yield os.path.join(d, f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--org", default=ORG)
    ap.add_argument("--execute", action="store_true", help="actually upload and update the registry")
    ap.add_argument("--private", action="store_true", help="create new repos as private")
    ap.add_argument("--names", help="optional JSON file {\"relative/path.h5\": \"RepoName\"} to set names yourself")
    ap.add_argument("--only", help="only process files whose name contains this text")
    args = ap.parse_args()

    paths = [p for p in find_models(args.root) if not args.only or args.only.lower() in os.path.basename(p).lower()]
    if not paths:
        sys.exit("No .h5/.keras files found.")

    overrides = {}
    if args.names:
        with open(args.names) as f:
            overrides = {k.replace("\\", "/"): v for k, v in json.load(f).items()}

    # Name = every folder on the way + file name, so the repo name says what the model is.
    parts = {p: os.path.relpath(p, args.root).replace("\\", "/").split("/") for p in paths}
    depth = {p: len(parts[p]) for p in paths}

    def build(p):
        rel = "/".join(parts[p])
        if rel in overrides:
            return repo_name(overrides[rel])
        segs = parts[p][-depth[p]:]
        segs = segs[:-1] + [os.path.splitext(segs[-1])[0]]
        return repo_name("_".join(segs))

    while True:
        groups = {}
        for p in paths:
            groups.setdefault(build(p), []).append(p)
        clashes = [g for g in groups.values() if len(g) > 1]
        if not clashes:
            break
        grew = False
        for g in clashes:
            for p in g:
                rel = "/".join(parts[p])
                if rel not in overrides and depth[p] < len(parts[p]):
                    depth[p] += 1
                    grew = True
        if not grew:
            sys.exit("Could not make names unique: " + ", ".join(g[0] for g in clashes))

    rows = [(p, build(p)[:96], os.path.basename(p)) for p in paths]

    print(f"{'LOCAL FILE':60} -> REPO / REGISTRY KEY")
    for p, name, fname in rows:
        print(f"{os.path.relpath(p, args.root):60} -> {args.org}/{name}  ({fname})")
    if not args.execute:
        print("\nDry run only. Re-run with --execute to upload.")
        return

    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    with open(REGISTRY) as f:
        registry = json.load(f)
    for p, name, fname in rows:
        repo_id = f"{args.org}/{name}"
        print(f"Uploading {fname} -> {repo_id}")
        api.create_repo(repo_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_file(path_or_fileobj=p, path_in_repo=fname, repo_id=repo_id, repo_type="model")
        registry[name] = {"repo_id": repo_id, "filename": fname}
        with open(REGISTRY, "w") as f:  # save after each model so a failure loses nothing
            json.dump(registry, f, indent=2)
    print("Done. Registry updated; test with load_model('<name>').summary()")


if __name__ == "__main__":
    main()
