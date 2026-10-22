"""Check every registry model WITHOUT downloading the weights.

Default: asks Hugging Face (metadata only) whether each repo + file exists.
Optional: --load NAME fully downloads and loads just that one model.

    python tools/check_models.py                      # check all, no downloads
    python tools/check_models.py --load ViT_TTE47_baseline
    python tools/check_models.py --load-all           # downloads everything (slow)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
REGISTRY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "echoforge", "config", "registry.json")


def check_remote(api, name, info):
    repo, fname = info["repo_id"], info.get("filename")
    if "PLACEHOLDER" in repo or "PLACEHOLDER" in (fname or ""):
        return "SKIP", "placeholder, not published yet"
    try:
        files = api.list_repo_files(repo)
    except Exception as e:
        return "FAIL", f"repo not reachable ({type(e).__name__}); private? set HF_TOKEN"
    if fname and info.get("subfolder"):
        fname = f"{info['subfolder']}/{fname}"
    if fname and fname not in files:
        return "FAIL", f"file '{fname}' not in repo (found: {', '.join(files[:4])})"
    return "OK", f"{repo}/{fname or '(whole repo)'}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", action="append", default=[], help="download + load this model (repeatable)")
    ap.add_argument("--load-all", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    with open(REGISTRY) as f:
        registry = json.load(f)

    bad = 0
    print(f"{'MODEL':50} STATUS  DETAIL")
    for name, info in registry.items():
        status, detail = check_remote(api, name, info)
        bad += status == "FAIL"
        print(f"{name:50} {status:6}  {detail}")

    to_load = list(registry) if args.load_all else args.load
    if to_load:
        from echoforge import load_model
    for name in to_load:
        try:
            m = load_model(name)
            print(f"LOAD OK    {name}  input={getattr(m, 'input_shape', '?')} output={getattr(m, 'output_shape', '?')}")
        except Exception as e:
            bad += 1
            print(f"LOAD FAIL  {name}: {type(e).__name__}: {str(e)[:200]}")

    print(f"\n{len(registry)} models checked, {bad} problem(s).")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
