import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types
from unittest.mock import patch

import pytest

from hust_prefill_microflow import activation as a


def test_default_import_and_registration_are_inert():
    code = "import sys; import hust_prefill_microflow as m; m.register(); assert not any(x in sys.modules for x in ['torch','torch_npu','betterscale']); print('inert')"
    env = dict(os.environ, BETTERSCALE_PREFILL_MICROFLOW="0",
               PYTHONPATH=str(Path(__file__).parents[1] / "src"))
    child = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert child.returncode == 0, child.stderr


def test_manifest_parses_with_current_manager():
    from vllm_hust_ext.manifest import parse_manifest
    manifest = a.document("vllm-hust-extension-v0.3.json")
    parsed = parse_manifest(manifest)
    assert manifest["schema_version"] == "0.3"
    assert parsed is not None
    assert len(manifest["resource_claims"]) == 3
    assert all(c["mode"] == "exclusive" for c in manifest["resource_claims"])
    assert manifest["activation"]["environment"][a.FLAG] == "1"


def test_extracted_default_off_and_pinned_maintainer():
    from hust_prefill_microflow.runtime.microflow import enabled
    assert not enabled({})
    metadata = json.loads((Path(__file__).parents[1] / "MOD_METADATA.json").read_text(encoding="utf8"))
    assert metadata["responsibility"]["maintainers"][0]["github"] == "CubeLander"
    assert metadata["evidence"]["qualification"] == "unqualified"


def test_donor_hashes_fail_closed(tmp_path):
    path = tmp_path / "betterscale" / "foo.py"
    path.parent.mkdir()
    path.write_bytes(b"original\n")
    pins = {"commit": "frozen", "files": {"betterscale/foo.py": hashlib.sha256(path.read_bytes()).hexdigest()}}
    assert a.verify_donor(tmp_path, pins)["source_files_verified"] == 1
    path.write_bytes(b"modified\n")
    with pytest.raises(ValueError, match="Unqualified donor"):
        a.verify_donor(tmp_path, pins)


def test_donor_pin_cannot_escape_root(tmp_path):
    with pytest.raises(ValueError, match="escaped"):
        a.verify_donor(tmp_path, {"commit": "x", "files": {"../escape.py": "0" * 64}})


def native_fixture(tmp_path):
    names = a.document("bridge-contract.json")["native_libraries"]
    raw = bytearray(64)
    raw[:6] = b"\x7fELF\x02\x01"
    raw[18:20] = (183).to_bytes(2, "little")
    for name in names:
        (tmp_path / name).write_bytes(raw)
    manifest = {"schema": "prefill-microflow-native-v1", "libraries": {n: hashlib.sha256(raw).hexdigest() for n in names}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return names, path, manifest


def test_native_identity_is_checked_without_loading(tmp_path):
    names, path, _ = native_fixture(tmp_path)
    assert len(a.verify_native(tmp_path, path, names)) == 2
    (tmp_path / names[0]).write_bytes(b"wrong architecture")
    with pytest.raises(ValueError, match="ELF64"):
        a.verify_native(tmp_path, path, names)


def test_native_content_change_is_rejected(tmp_path):
    names, path, _ = native_fixture(tmp_path)
    with (tmp_path / names[0]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="content mismatch"):
        a.verify_native(tmp_path, path, names)


def test_missing_or_extra_native_identity_is_rejected(tmp_path):
    names, path, manifest = native_fixture(tmp_path)
    manifest["libraries"]["unexpected.so"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="exactly"):
        a.verify_native(tmp_path, path, names)


def test_off_on_caches_are_disjoint_and_wrong_cache_rejected(tmp_path):
    off, on = (a.cache_environment(tmp_path, mode) for mode in ("off", "on"))
    a.verify_caches(on)
    for name in a.document("bridge-contract.json")["cache_variables"]:
        assert off[name] != on[name]
        broken = on | {name: off[name]}
        with pytest.raises(ValueError, match="cache separation"):
            a.verify_caches(broken)


def test_unmanaged_opt_in_is_rejected_without_host_import():
    with patch.dict(os.environ, {a.FLAG: "1", a.MANAGED: "0"}):
        with pytest.raises(ValueError, match="ECPA"):
            a.register()


def test_bad_switch_is_rejected():
    with patch.dict(os.environ, {a.FLAG: "true"}):
        with pytest.raises(ValueError, match="0 or 1"):
            a.register()


def test_loaded_donor_implementation_cannot_be_replaced():
    contract = a.document("bridge-contract.json")
    key = contract["alias_prefix"] + contract["modules"][0]
    with patch.dict(sys.modules, {key: types.ModuleType(key)}):
        with pytest.raises(ValueError, match="imported before"):
            a.install_bridge(contract)


def test_alias_bridge_loads_actual_independent_host_only_plan(tmp_path):
    # A fake parent package exposes only the module-loading seam, not a device.
    prefix = "microflow_cpu_fixture."
    package = types.ModuleType(prefix[:-1])
    package.__path__ = [str(tmp_path)]
    contract = {"alias_prefix": prefix, "implementation_prefix": "hust_prefill_microflow.runtime.", "modules": ["microflow_plan"]}
    with patch.dict(sys.modules, {prefix[:-1]: package}):
        finder = a.install_bridge(contract)
        try:
            loaded = importlib.import_module(prefix + "microflow_plan")
            plans = loaded.slabs(4096, [4096], [False])
            assert tuple(p.tokens for p in plans) == (2048, 2048)
            assert a.install_bridge(contract) is finder
            assert finder.find_spec("not_a_microflow_module") is None
        finally:
            sys.meta_path.remove(finder)


def test_restart_required_for_rollback_or_activation_change():
    with patch.object(a, "_active_signature", ("already-active",)):
        with patch.dict(os.environ, {a.FLAG: "0"}):
            with pytest.raises(RuntimeError, match="fresh worker"):
                a.register()
        with patch.dict(os.environ, {a.FLAG: "1", a.MANAGED: "1"}):
            with pytest.raises(RuntimeError, match="restart"):
                a.register()


def test_all_extracted_file_hashes_match_provenance():
    root = Path(__file__).parents[1]
    provenance = json.loads((root / "provenance.json").read_text(encoding="utf8"))
    for item in provenance["files"]:
        assert hashlib.sha256((root / item["extracted_path"]).read_bytes()).hexdigest() == item["extracted_sha256"]
