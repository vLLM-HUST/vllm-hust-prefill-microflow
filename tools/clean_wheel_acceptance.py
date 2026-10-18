"""Verify an installed wheel and ECPA discovery in isolated child interpreters."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
bundle = "org.vllm-hust.prefill-microflow"
commands = []
with tempfile.TemporaryDirectory(prefix="microflow-ecpa-") as directory:
    env = dict(os.environ, VLLM_HUST_EXT_CONFIG=str(Path(directory) / "intent.json"),
               BETTERSCALE_PREFILL_MICROFLOW="0", HUST_PREFILL_MICROFLOW_ENABLED="0")
    code = """import json,sys,importlib.metadata as m
import hust_prefill_microflow as plugin
plugin.register()
from vllm_hust_ext.discovery import discover_bundles
rows=discover_bundles(['org.vllm-hust.prefill-microflow'])
assert len(rows)==1
assert not any(k in sys.modules for k in ('torch','torch_npu','betterscale'))
assert 'site-packages' in plugin.__file__.replace('\\\\','/')
print(json.dumps(dict(installed_version=m.version('vllm-hust-prefill-microflow'),static_bundle=rows[0].bundle_id,no_runtime_imports=True,isolated_interpreter=bool(sys.flags.isolated))))
"""
    child = subprocess.run([sys.executable, "-I", "-c", code], env=env, capture_output=True, text=True)
    assert child.returncode == 0, child.stderr
    installed = json.loads(child.stdout)
    for action in ("inspect", "validate", "enable", "check", "plan", "disable", "forget"):
        command = [sys.executable, "-I", "-c",
                   "from vllm_hust_ext.cli import main; raise SystemExit(main())",
                   "extension", action, bundle]
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        commands.append({"action": action, "exit_code": result.returncode,
                         "stdout": result.stdout, "stderr": result.stderr})
        assert result.returncode == 0, commands[-1]
        if action == "check":
            states = json.loads(result.stdout)["states"]
            assert "degraded" in states and "runtime_effective" not in states
    # Intent/plan may succeed for an unknown host. The actual callback must fail
    # before any numerical implementation is imported, rather than inventing a
    # CLI exit-code guarantee that the Manager does not provide.
    blocked_code = """import json,os,sys
from hust_prefill_microflow.activation import cache_environment,register
os.environ.update(cache_environment(sys.argv[1], 'on'))
try:
    register()
except ValueError as exc:
    assert 'host distribution' in str(exc) or 'Unqualified' in str(exc)
    assert not any(k in sys.modules for k in ('torch','torch_npu','betterscale'))
    print(json.dumps(dict(unknown_host_blocked=True,reason=str(exc),no_runtime_imports=True)))
else:
    raise AssertionError('Unknown clean host was allowed to activate')
"""
    blocked = subprocess.run([sys.executable, "-I", "-c", blocked_code, directory],
                             env=env, capture_output=True, text=True)
    assert blocked.returncode == 0, blocked.stderr
    runtime_guard = json.loads(blocked.stdout)
report = {"schema": "microflow-clean-wheel-cpu-v1", "installed": installed,
          "manager_version": __import__("importlib.metadata").metadata.version("vllm-hust-ext"),
          "commands": commands, "runtime_guard": runtime_guard,
          "new_models": 0, "device_validation": False,
          "whole_model_quality_or_cost_qualified": False,
          "scope": "Installed-wheel discovery/validation, enable intent, degraded unknown-host status, plan and disable/forget; actual callback rejects unknown host without numerical imports. No model/run or live-device qualification."}
args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8", newline="\n")
print(json.dumps({"installed": installed, "cli_checks": len(commands), "device_validation": False}))
