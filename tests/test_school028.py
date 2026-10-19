import hashlib
import json
from pathlib import Path
import subprocess
import sys

from hust_prefill_microflow import school028


def test_school_discovery_is_inert():
    result = subprocess.run([sys.executable, "-c", "import sys; from hust_prefill_microflow.school028 import register; register(); assert 'torch' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_school_manifest_and_source_contract():
    from vllm_hust_ext.manifest import parse_manifest
    root = Path(school028.__file__).parent
    manifest = json.loads((root / "vllm-hust-extension-v0.3.json").read_text())
    assert parse_manifest(manifest)
    assert manifest["host"]["version_range"] == "==0.28.1.post1.dev143"
    assert len(manifest["resource_claims"]) == 1
    contract = json.loads((root / "host-contract.json").read_text())
    assert "gf18cf803c" in contract["versions"]["vllm"]
    assert len(contract["sources"]["betterscale"]) > 100
    for files in contract["sources"].values():
        for relative, digest in files.items():
            assert not Path(relative).is_absolute()
            assert ".." not in Path(relative).parts
            assert len(bytes.fromhex(digest)) == hashlib.sha256().digest_size


def test_original_host_contract_is_not_widened():
    from hust_prefill_microflow.activation import document
    assert document("host-contract.json")["versions"]["vllm"] == "0.25.1"
