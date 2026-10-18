"""Explicit, fail-closed bridge to the pinned donor publication seam.

No tensor code, native library, monkey patch or worker is loaded during discovery.
The import redirects below are deliberate runtime hooks and are declared as
exclusive resources in the Manifest. They do not rewrite installed donor files.
"""
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

PACKAGE = Path(__file__).parent
FLAG = "BETTERSCALE_PREFILL_MICROFLOW"
MANAGED = "HUST_PREFILL_MICROFLOW_ENABLED"
_active_signature = None


def document(name):
    return json.loads((PACKAGE / name).read_text(encoding="utf8"))


def verify_donor(root, pins):
    root = Path(root).resolve()
    for relative, expected in pins["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Donor pin escaped its installation root")
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Unqualified donor source: " + relative)
    return {"commit": pins["commit"], "source_files_verified": len(pins["files"])}


def verify_native(root, manifest_file, names):
    root = Path(root).resolve()
    manifest = json.loads(Path(manifest_file).read_text(encoding="utf8"))
    if set(manifest) != {"schema", "libraries"} or manifest["schema"] != "prefill-microflow-native-v1":
        raise ValueError("Unknown native identity contract")
    if not isinstance(manifest["libraries"], dict) or set(manifest["libraries"]) != set(names):
        raise ValueError("Native identity must cover exactly KKT and WU")
    result = {}
    for name in names:
        expected = manifest["libraries"][name]
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Invalid native SHA256: " + name)
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Missing or external native library: " + name)
        raw = path.read_bytes()
        if len(raw) < 64 or raw[:6] != b"\x7fELF\x02\x01" or int.from_bytes(raw[18:20], "little") != 183:
            raise ValueError("Expected an ELF64 little-endian aarch64 library: " + name)
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected:
            raise ValueError("Native content mismatch: " + name)
        result[name] = actual
    return result


def cache_environment(root, arm):
    if arm not in ("off", "on"):
        raise ValueError("Cache arm must be off or on")
    base = Path(root).resolve()
    result = {
        "HUST_PREFILL_MICROFLOW_CACHE_ROOT": str(base),
        "HUST_PREFILL_MICROFLOW_ARM": arm,
        FLAG: "1" if arm == "on" else "0",
        MANAGED: "1" if arm == "on" else "0",
    }
    for name, leaf in document("bridge-contract.json")["cache_variables"].items():
        result[name] = str(base / arm / leaf)
    return result


def verify_caches(env):
    if not env.get("HUST_PREFILL_MICROFLOW_CACHE_ROOT"):
        raise ValueError("Declare a task-owned OFF/ON cache root")
    expected = cache_environment(env["HUST_PREFILL_MICROFLOW_CACHE_ROOT"], "on")
    for name, value in expected.items():
        if name.endswith("CACHE_ROOT") or name.endswith("CACHE_DIR"):
            if not env.get(name) or Path(env[name]).resolve() != Path(value).resolve():
                raise ValueError("OFF/ON compile-cache separation failed: " + name)
        elif env.get(name) != value:
            raise ValueError("Invalid activation arm: " + name)


def preflight(env=None):
    """Metadata/file checks only; no device initialization or library loading."""
    env = os.environ if env is None else env
    verify_caches(env)
    pins = document("donor-pins.json")
    # Runtime identity is checked as well as the full donor source content.
    host = document("host-contract.json")
    for distribution, expected in host["versions"].items():
        try:
            actual = importlib.metadata.version(distribution).split("+", 1)[0]
        except importlib.metadata.PackageNotFoundError as exc:
            raise ValueError("Missing qualified host distribution: " + distribution) from exc
        if actual != expected:
            raise ValueError(f"Unqualified {distribution}: expected {expected}, got {actual}")
    spec = importlib.util.find_spec("betterscale")
    if spec is None or not spec.origin:
        raise ValueError("Install the pinned, separately qualified BetterScale shared host")
    donor = verify_donor(Path(spec.origin).parent.parent, pins)
    for name in ("BETTERSCALE_MICROFLOW_NATIVE_DIR", "HUST_PREFILL_MICROFLOW_NATIVE_MANIFEST"):
        if not env.get(name):
            raise ValueError("Missing explicit deployment value: " + name)
    native = verify_native(env["BETTERSCALE_MICROFLOW_NATIVE_DIR"],
                           env["HUST_PREFILL_MICROFLOW_NATIVE_MANIFEST"],
                           document("bridge-contract.json")["native_libraries"])
    return {"donor": donor, "native": native, "device_qualified": False,
            "quality_qualified": False, "runtime_effective": False}


class AliasLoader(importlib.abc.Loader):
    def __init__(self, target):
        self.target = target

    def create_module(self, spec):
        return importlib.import_module(self.target)

    def exec_module(self, module):
        return None


class MicroflowFinder(importlib.abc.MetaPathFinder):
    def __init__(self, contract):
        self.aliases = {contract["alias_prefix"] + name:
                        contract["implementation_prefix"] + name for name in contract["modules"]}

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.aliases:
            return None
        return importlib.machinery.ModuleSpec(fullname, AliasLoader(self.aliases[fullname]))


def install_bridge(contract):
    existing = next((x for x in sys.meta_path if isinstance(x, MicroflowFinder)), None)
    candidate = MicroflowFinder(contract)
    if existing:
        if existing.aliases != candidate.aliases:
            raise ValueError("A different Microflow bridge is already installed; restart workers")
        return existing
    if any(name in sys.modules for name in candidate.aliases):
        raise ValueError("Donor Microflow was imported before ECPA selection; restart workers")
    sys.meta_path.insert(0, candidate)
    return candidate


def register():
    global _active_signature
    value = os.environ.get(FLAG, "0")
    if value not in ("0", "1"):
        raise ValueError(FLAG + " must be 0 or 1")
    if value == "0":
        if _active_signature is not None:
            raise RuntimeError("Disable requires a fresh worker; in-process rollback is unsupported")
        return None
    if os.environ.get(MANAGED) != "1":
        raise ValueError("Select and enable this bundle through ECPA before starting workers")
    contract = document("bridge-contract.json")
    signature = tuple(sorted((name, os.environ.get(name)) for name in
                            [FLAG, MANAGED, "BETTERSCALE_MICROFLOW_NATIVE_DIR",
                             "HUST_PREFILL_MICROFLOW_NATIVE_MANIFEST",
                             "HUST_PREFILL_MICROFLOW_CACHE_ROOT", "HUST_PREFILL_MICROFLOW_ARM",
                             *contract["cache_variables"]]))
    if _active_signature is not None:
        if signature != _active_signature:
            raise RuntimeError("Activation changed after registration; restart workers")
        return None
    preflight()
    install_bridge(contract)
    _active_signature = signature
    # This proves only registration, never execution, quality or performance.
    return None
