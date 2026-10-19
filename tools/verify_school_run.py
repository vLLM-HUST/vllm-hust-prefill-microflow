"""Verify a bounded MMLU FA screen; never infer task accuracy from completion."""

import argparse
import hashlib
import json
from pathlib import Path


def verify(root, run):
    directory = root / run
    receipt = json.loads((directory / "receipt.json").read_text())
    assert receipt["status"] == "measured"
    assert receipt["all_devices_released"]
    # Manager returns 130 after the controller's intentional SIGINT shutdown.
    assert receipt["child_exit_codes"] in ([0, 0], [0, 130])
    if receipt["child_exit_codes"][-1] == 130:
        log = (directory / "service.log").read_text()
        assert "request processing complete; starting resource teardown" in log
    for name, digest in receipt["input_hashes"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    raw = json.loads((directory / "mmlu-pro.json").read_text())
    assert raw["completed"] == 200 and raw["failed"] == 0
    assert raw["total_output_tokens"] == 51200
    assert raw["total_input_tokens"] == 282265
    assert set(raw["output_lens"]) == {256}
    control = json.loads((root / "mmlu-b1-kvmat-e24-r36-screen-r1/mmlu-pro.json").read_text())
    assert raw["input_lens"] == control["input_lens"]
    command = json.loads((directory / "launch-command.json").read_text())
    for flag, value in {
        "--max-num-seqs": "24", "--max-model-len": "65536",
        "--max-num-batched-tokens": "4096",
        "--kv-cache-memory-bytes": "21474836480",
        "--tensor-parallel-size": "2",
    }.items():
        assert command[command.index(flag) + 1] == value
    policy = [json.loads(line) for line in (directory / "kvmat-policy.jsonl").read_text().splitlines()]
    engine = [json.loads(line) for line in (directory / "kvmat-engine.jsonl").read_text().splitlines()]
    admitted = [e for e in engine if e.get("event") == "school028_request_salt_admitted"]
    assert len(policy) == len(admitted) == 200
    assert all(e["runtime_effective_decision"] == "recompute" for e in policy)
    assert {hashlib.sha256(e["runtime_cache_salt"].encode()).hexdigest() for e in policy} == {e["cache_salt_sha256"] for e in admitted}
    active = receipt["measurement_role"].endswith(" on")
    ranks = []
    for path in sorted(directory.glob("microflow-*.jsonl")):
        events = [json.loads(line) for line in path.read_text().splitlines()]
        loaded = [e for e in events if e["event"] == "model_loaded"]
        if not loaded:
            continue
        assert len(loaded) == 1 and loaded[0]["active"] == active
        assert not loaded[0]["gdn_modified"] and not loaded[0]["draft_modified"]
        captures = [e for e in events if e["event"] == "prefix" and e["capturing"] and e["split"]]
        checks = [e for e in events if e["event"] == "partition_equivalence"]
        replays = [e for e in events if e["event"] == "graph_replay"
                   and receipt["bench_started_wall"] <= e["time"] <= receipt["bench_ended_wall"]
                   and e.get("target") is True and not e["draft"]
                   and e["tokens"] in (1024, 2048, 4096, 8192)]
        if active:
            assert {e["tokens"] for e in checks} == {1024, 2048, 4096}
            assert all(all(e["exact"]) for e in checks)
            assert len({e["layer"] for e in captures}) == 10
            assert not any(e["layer"].startswith("mtp.") for e in captures)
            assert replays
            assert {e["tokens"] for e in replays} <= {e["tokens"] for e in captures}
        else:
            assert not captures and not checks
        ranks.append({"rank": loaded[0]["rank"], "split_shape_replays": len(replays),
                      "captured_target_layers": len({e["layer"] for e in captures})})
    assert {r["rank"] for r in ranks} == {0, 1}
    result = {
        "run_id": run, "active": active, "scope": "FA-only service performance screen",
        "quality_rescored": False, "source_commit": receipt["microflow_source_commit"],
        "requests": raw["completed"], "failures": raw["failed"],
        "kv_policy_engine_admissions": len(admitted),
        "shutdown_exit_code": receipt["child_exit_codes"][-1],
        "output_tps": raw["total_output_tokens"] / raw["duration"],
        "change_vs_current_b1_pct": 100 * (raw["output_throughput"] / control["output_throughput"] - 1),
        "bounded_text_matches_vs_current_b1": sum(a == b for a, b in zip(raw["generated_texts"], control["generated_texts"])),
        "ranks": ranks,
    }
    assert abs(result["output_tps"] - raw["output_throughput"]) < 1e-8
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("run")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root, args.run)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
