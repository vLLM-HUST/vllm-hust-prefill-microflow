"""Generate a candidate source contract; generation alone is not admission."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


def freeze(site, donor, output):
    versions = {d.metadata["Name"].lower().replace("_", "-"): d.version
                for d in importlib.metadata.distributions(path=[str(site)])}
    sources = {"betterscale": {
        str(p.relative_to(donor)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(donor.rglob("*")) if p.suffix in (".py", ".json")}}
    for package, names in {
        "vllm": ["model_executor/models/qwen3_5.py", "model_executor/models/qwen3_next.py"],
        "vllm_ascend": ["patch/worker/patch_qwen3_5.py", "compilation/acl_graph.py",
                        "ops/triton/linearnorm/split_qkv_rmsnorm_mrope.py"],
    }.items():
        sources[package] = {name: hashlib.sha256((site / package / name).read_bytes()).hexdigest()
                            for name in names}
    output.write_text(json.dumps({
        "scope": "experimental FA-only; no GDN or draft replacement",
        "versions": {name: versions[name] for name in
                     ["vllm", "vllm-ascend", "torch", "torch-npu", "transformers"]},
        "sources": sources,
    }, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("site", type=Path)
    p.add_argument("donor", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    freeze(args.site, args.donor, args.output)
