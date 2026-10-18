# Integrate prefill overlap without losing the gains at the boundary

Use when changing `BETTERSCALE_PREFILL_MICROFLOW`, its metadata banks, native
KKT/WU launch contract, or comparing standalone pipeline timing to serving.
This is the GDN + FA integration branch, not MLP/AllReduce or conv fusion.

## Observed component gates (2026-10-18)

ascend180 physical4 (container visibility1, child torch device0), 910B2/CANN9.1,
torch2.10+cpu / torch_npu2.10.post4 / Triton3.5.0. Both cooperative locks,
30-second idle admission and idle release; unrelated tasks preserved. The
previous prototype host had Triton3.2: compare each arm on this same host.

- FA run04: actual registered prefix custom op, eager with/without residual,
  then three graph replays changing input/residual/positions. All five outputs
  exactly equal the unsplit donor oracle. Four equal chunks; full48 Vector.
  Median4K618.31→518.78µs,8K1201.00→949.07µs (12 alternating samples, three
  calls/graph). This **excludes attention and the layer tail**, and leaves KV
  scatter to the original attention call. It is not a whole-model speedup.
- GDN run13: projection through joined H/O/core output and z; excludes output
  norm/projection. Single4K, two2K, ragged4K, mixed verification and shrinking
  to120 live tokens. Original64-token request-relative chunk boundaries are
  preserved even when splitting a sequence. Outputs, whole convolution
  history and the whole recurrent pool are exactly equal in eager and changed
  metadata/input/state graph replays. Verification accepted counts include1/2/3.
  Latest single4K medians2310.48→2334.21µs: **approximately flat, not a demonstrated
  GDN production speedup**. Do not substitute the older serialized two-slab
  prototype denominator for this unsplit original implementation.

### Final component boundary and production-shaped projections

Run14 repeats the gates with `F.linear` and donor-style row-major `[N,K]`
weights, rather than the initial harness's `[K,N]` MM. FA also tests the48-token
fallback, and GDN includes an additional donor gated RMSNorm + local output
projection gate. All eager/replay outputs and complete GDN states remain exact.

| Window | Original µs | Microflow µs |
|---|---:|---:|
| FA prefix4K |618.77|525.21|
| FA prefix8K |1202.27|967.67|
| GDN core + z4K |2303.07|2323.24|
| GDN through local output projection4K |2537.49|2510.84|

FA reductions are15.1%/19.5% in this component window. GDN remains approximately
flat; the complete local chain's1.05% median reduction is not a robust production
speedup claim. The old strided z requires a materializing reshape at its norm
consumer; Microflow's already-contiguous z does not. Comparing only the producer
window misses that downstream copy. TP AllReduce, attention and whole-model
serving are not included in these component timings.

### Dynamic compiler boundary

The first whole-model ON launch (run12) was correctly rejected by Dynamo:
`input_ids.size(0)` is explicitly marked dynamic, but the outer Python
`tokens == 4096` branch specialized it. Move token-count selection **inside**
the opaque GDN/FA custom ops, with the original small-shape operator sequence
as their fallback. Do not disable compilation or change vLLM's dynamic contract.
Keep shared rotary-row selection outside the FA op and pass the selected tensor
in, so model-wide sharing remains visible to the outer graph.

CPU `test_qwen35_microflow_dynamic.py` now marks token0 dynamic and checks
4096→48→128 in one full Dynamo graph for each adapter. The actual donor projection
class is `AscendQKVParallelLinear`, not just `QKVParallelLinear`; both are admitted,
while projection variants carrying a custom sequence-parallel op are left alone.
The admission regression test protects against an apparently enabled but silently
inactive FA path.

Run12's OFF service completed its six real-weight requests. ON failed before
serving; its API parent stayed alive after the fatal worker error. Explicitly
stopping the owned controller released both devices. Future controllers must
recognize fatal worker startup errors rather than waiting only for API-parent
exit. OFF remains a valid reference because subsequent corrections only change
the flag-enabled branch. Do not call the failed ON launch a successful smoke.

The reusable probes require the existing qualified runtime, not an installer.
After acquiring a free selected-device lease, set `CAPSULE` to a fresh external
output directory, `TASK_QUEUE_ENABLE=0`, `MTP_TOKENS=2`, the existing GDN H/O
library paths, and for GDN also `BETTERSCALE_PREFILL_MICROFLOW=1` plus the two
native Microflow libraries. Build instructions are in the package's
`patches/qwen_gdn/microflow_native/README.md`. `probe_fa.py` and `probe_gdn.py`
are synthetic-weight component gates; they do not claim model-level admission.

## Integration costs and traps that changed the implementation

- Initial dynamic integration passed exact state gates but regressed GDN:
  run07 unsplit2308.59 versus2518.11µs. Trace run08 exposes additional input
  gathers, repeated layout copies and idle-row work. Correctness alone was not
  enough to ship the claimed optimization.
- Fixed-capacity index arrays have a trailing empty sentinel suffix. Native WU
  was still doing zero matrix products for it. Trim that suffix **on device,
  identically for all Cube/Vector participants before priming flags**; keep
  the compiled1:1 KKT /1:2 WU ratio intact. Run09 reduced candidate latency to
  2411.25µs, preserving changing-metadata and empty-second-slab gates.
- Join Q/K/W/U/g and raw-order z in one routing pass, rather than launching a
  separate per-token kernel for each. Slab0 is already a contiguous input
  prefix; its inactive tail need not be gathered. Run10 candidate2332.03µs.
- Materialize the native convolution's transposed weight once per call rather
  than once per role/slab on contended queues. Run11 candidate2321.03µs; the
  later accepted-count gate remains approximately flat. These small differences
  are not a reason to claim a robust GDN win.
- MTPFrame rebinds top-level tensors to banked views after Core construction.
  Pipeline field lookups must resolve the **current Core tensors**, never cache
  the old device pointers. Routing is published per wave; graph keys do not
  encode request lengths or roles. Never read live metadata back to CPU during
  the captured path.
- Native scratch and masks must explicitly be FP32, regardless of the model
  loader's default dtype. Native source/API interpretation is not inferred
  from a Torch tensor's default allocation dtype.
- FA runs01–03 failed because the TEST sampled positions up to32767 then added1,
  outside a32768-row rotary table. Event retention and record_stream changes
  were not the cause or fix; both speculative changes were reverted. Run04
  limits initial positions to29999 and passes with the original stream logic.
  Inspect the first device assertion, not just the final synchronization error.

Raw controllers, exact source snapshots, native builds, failed/successful
receipts and traces belong in the external evidence capsule
`/root/lumi-microflow-integration-20261018`, not Git or the installed runtime.

## Small FA plans and numerical attribution

The sibling search's fixed TP2 table is1024→512×2,2048→1024×2,
4096→1024×4,8192→2048×4. Its separate prefix includes KV scatter;
our integration leaves scatter inside the original attention consumer. Do not
copy those percentages onto this integrated window or whole-model service.
Search evidence: `/Users/fletcher-tian/Archives/BetterScale/fa-small-batch-20261018/report.html`.

Run17's exact UNSPLIT comparison failed at1K. Run20 isolates the source instead
of relaxing tolerances: feed identical normalized input bytes into serial
`F.linear` whole1K versus512×2, using donor-style `[N,K]` BF16 weights.
Input Norm full/split is exact; QKV GEMM differs at902 elements, max abs0.015625,
relative L2 3.2222e-5. At2K versus1024×2 the same control is exact. This proves
an observed M-shape-dependent GEMM numerical effect, not its internal cause;
it does NOT establish that every split changes accumulation or explain unrelated
decode divergence. Splitting M leaves the mathematical dot products unchanged.

Both small integrated schedules equal the SAME-PARTITION serial donor exactly,
in eager (with/without residual) and three changed-input/position graph replays.
The probe preserves both oracles and writes separate normalization/GEMM attribution.
No arbitrary floating-point tolerance replaces the exact scheduling check.
Run20 unprofiled medians1K199.76→198.59µs (roughly flat),2K344.33→317.25µs.
These are single-rank component measurements while the model gate uses other
cards; do not reinterpret them as the sibling search's controlled performance.

Whole-model run16 and run18 initially expose a later decode token divergence
for one4K request, despite exact first-token logprob. Repeated OFF reproduces
all six original token sequences. Newly enabled1K also changes first-token
logprob and later tokens. These are not automatically accepted as harmless
roundoff. MTP draft was unnecessarily wrapped in the FA custom op; target-only
admission restores its original compiler boundary. End-to-end qualification
requires the subsequent controlled result, not just the component checks above.

### Target-only integration handoff

Run21 excludes `mtp.*` layers statically. The generated rank0 `eagle_head`
computation graph contains no Microflow ops (receipt `draft-boundary.json`).
45 CPU tests pass, including marked-dynamic4096→48→128→1024→2048→8192 with
one FX graph and a draft-admission regression. The eight-request real-weight
BF16/TP2/MTP2 service compiles, captures, serves and releases both cards normally.
All first token IDs match OFF; first logprob is exact except1K (delta−0.08268).
Full16-token sequences match5/8: differences start at zero-based token7 for
one4K prompt,6 for2049,3 for1024. Repeated OFF matches the earlier six requests.
**Draft exclusion does not eliminate the target-generation difference.**
No claim that all these differences are caused by GEMM shape rounding, or that
component parity establishes end-to-end quality. Fletcher chose to stop digging
into GEMM internals and keep draft untouched; no accuracy campaign or further
schedule search was launched. The flag remains default-off/experimental.

Local external evidence:
`/Users/fletcher-tian/Archives/BetterScale/prefill-microflow-20261018/evidence.tgz`.
Contains executed source snapshots, failed gates as well as passing ones, native
payloads, probes, model receipts, relevant generated graphs and release records.
Run19 was denied the physical4 cooperative lock while run17 was releasing; it
never ran a probe. Run20 admitted only after the lock was free. No foreign job
was interrupted and no installed runtime was modified.
