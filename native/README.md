# Prefill Microflow native leaves

These two libraries implement the existing experimental GDN KKT 24/24 and
WU 24/48 leaves. They are not a replacement for CANN or the existing GDN H/O
libraries. Build only inside the already-qualified Ascend910B2 / CANN9.1
runtime, against the donor's pinned Catlass headers; do not upgrade a runtime
or fetch an unpinned dependency as part of enabling the switch.

```sh
cmake -S native \
  -B "$EXTERNAL_BUILD_DIR" -DASCEND_CANN_PACKAGE_PATH="$QUALIFIED_CANN_ROOT" \
  -DCATLASS_ROOT="$PINNED_CATLASS_ROOT"
cmake --build "$EXTERNAL_BUILD_DIR" -j2
export BETTERSCALE_MICROFLOW_NATIVE_DIR="$EXTERNAL_BUILD_DIR/lib"
export BETTERSCALE_PREFILL_MICROFLOW=1
```

`EXTERNAL_BUILD_DIR` must be a task-owned directory outside the source tree.
The two required files are `libbs_microflow_kkt.so` and `libbs_microflow_wu.so`.
No compilation is performed during inference. The default search directory
is this package directory, for distributions that explicitly include the
qualified binaries. Missing binaries fail startup when the switch is enabled;
with the default switch value `0`, they are neither loaded nor required.

The KKT producer/consumer ratio is part of the compiled kernel, not inferred
from a stream's Vector quota. WU deliberately retains two Vector participants
per Cube. Publication padding uses empty sequence sentinels. Every native participant
trims the same trailing empty indices on device before priming its flags, so
captured fixed-capacity metadata does not cause work on idle request slots.
Remaining partial tiles preserve the native cross-core handshake.

Source provenance: BetterScale's `kkt-ratio` experiment; its included
`block_mmad_pingpong_tla_multi.hpp` retains the Tianjin University BSD-3-Clause
notice. Catlass headers are external CANN2.0 components, pinned by the existing
Qwen donor build (41bf90da655bba3c66d0acd7e00abe33960ecfd6). Applicable license
texts are shipped under the repository's `licenses/` directory. Do not strip
per-file notices when building or distributing the native payloads.
