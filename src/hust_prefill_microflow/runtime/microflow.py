# SPDX-License-Identifier: Apache-2.0
"""Opt-in prefill execution policy. Decode and the disabled path stay unchanged."""

import os

FLAG = "BETTERSCALE_PREFILL_MICROFLOW"


def enabled(env=None):
    value = (os.environ if env is None else env).get(FLAG, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{FLAG} must be 0 or 1; restart workers to change it")
    return value == "1"


def partition(tokens):
    # Fixed TP2 plans from the bounded FA prefix search, not a live search.
    # Called inside the opaque op: outer Dynamo token dimensions stay dynamic.
    pieces = {1024: 2, 2048: 2, 4096: 4, 8192: 4}.get(tokens)
    if pieces is None:
        return ()
    return tuple(tokens * i // pieces for i in range(pieces + 1))


def install():
    if not enabled():
        return
    from .microflow_attention import install as install_attention

    from .microflow_native import directory
    for name in ("libbs_microflow_kkt.so", "libbs_microflow_wu.so"):
        path = directory() / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing Microflow native binary: {path}")
    from .microflow_gdn import install as install_gdn

    install_attention()
    install_gdn()
