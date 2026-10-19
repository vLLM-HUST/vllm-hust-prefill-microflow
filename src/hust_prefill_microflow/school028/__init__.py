"""Default-off, exact-source school 0.28 FA-only experiment."""

import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import time


FLAG = "HUST_MICROFLOW_SCHOOL028_FA"


def event(kind, **fields):
    root = Path(os.environ["HUST_MICROFLOW_EVENT_DIR"])
    with (root / f"microflow-{os.getpid()}.jsonl").open("a") as stream:
        stream.write(json.dumps({"event": kind, "time": time.time(), **fields}) + "\n")


def preflight():
    contract = json.loads(Path(__file__).with_name("host-contract.json").read_text())
    for name, expected in contract["versions"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError("Unqualified school runtime: " + name)
    for package, files in contract["sources"].items():
        spec = importlib.util.find_spec(package)
        if spec is None or spec.origin is None:
            raise ValueError("Missing host package: " + package)
        root = Path(spec.origin).parent
        for relative, expected in files.items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("Host pin escapes package")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("Unqualified host source: " + relative)
    arm = os.environ.get("HUST_MICROFLOW_ARM")
    if arm not in ("off", "on"):
        raise ValueError("Explicit OFF/ON arm required")
    root = Path(os.environ["HUST_MICROFLOW_CACHE_ROOT"]).resolve()
    for name, leaf in [("VLLM_CACHE_ROOT", "vllm"), ("TRITON_CACHE_DIR", "triton"),
                       ("TORCHINDUCTOR_CACHE_DIR", "inductor")]:
        if Path(os.environ[name]).resolve() != root / arm / leaf:
            raise ValueError("Non-isolated compiler cache: " + name)
    return contract


def register():
    flag = os.environ.get(FLAG, "0")
    if flag not in ("0", "1"):
        raise ValueError("School FA flag must be 0 or 1")
    if flag == "0":
        return
    if os.environ.get("BETTERSCALE_PREFILL_MICROFLOW", "0") != "0":
        raise ValueError("Original Microflow and school FA adapter are exclusive")
    preflight()
