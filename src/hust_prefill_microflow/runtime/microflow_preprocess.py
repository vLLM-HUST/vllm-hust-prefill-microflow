# SPDX-License-Identifier: Apache-2.0
# Normalization/gating arithmetic follows the pinned vLLM FLA and Ascend donor.
"""Packed conv output -> Q/K norm, V and gates without intermediate cat/copies.

Keep the donor BF16 Q/K and beta rounding boundaries; recurrence is unchanged.
Explicit token/head axes avoid Ascend scalar gather lowering for head % 16.
The opt-in Q layout retains logical [1,T,8,128] but is physically head-major;
K stays token-major for the pinned KKT/WY consumers.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def preprocess_kernel(
    X,
    A,
    B,
    LOG,
    BIAS,
    Q,
    K,
    V,
    G,
    BETA,
    TOKEN_MAP,
    MAPPED: tl.constexpr,
    Q_HEAD_MAJOR: tl.constexpr,
    T: tl.constexpr,
    XS: tl.constexpr,
    AS: tl.constexpr,
    BS: tl.constexpr,
    R: tl.constexpr,
    CORES: tl.constexpr,
):
    for tile in range(tl.program_id(0), tl.cdiv(T, R), CORES):
        token = tile * R + tl.arange(0, R)
        source = tl.load(TOKEN_MAP + token, token < T, other=0) if MAPPED else token
        d = tl.arange(0, 128)[None, None, :]
        head = tl.arange(0, 8)[None, :, None]
        off = source[:, None, None] * XS + head * 128 + d
        mask = token[:, None, None] < T
        qx = tl.load(X + off, mask, other=0).to(tl.float32)
        kx = tl.load(X + off + 1024, mask, other=0).to(tl.float32)
        qy = qx * tl.rsqrt(tl.sum(qx * qx, 2) + 1.0e-6)[:, :, None]
        ky = kx * tl.rsqrt(tl.sum(kx * kx, 2) + 1.0e-6)[:, :, None]
        dest = token[:, None, None] * 1024 + head * 128 + d
        qdest = (head * T + token[:, None, None]) * 128 + d if Q_HEAD_MAJOR else dest
        tl.store(Q + qdest, qy, mask)
        tl.store(K + dest, ky, mask)
        vd = tl.arange(0, 4096)[None, :]
        vm = (token[:, None] < T) & (vd < 2048)
        value = tl.load(X + source[:, None] * XS + 2048 + vd, vm, other=0)
        tl.store(V + token[:, None] * 2048 + vd, value, vm)
        gh = tl.arange(0, 32)[None, :]
        gm = (token[:, None] < T) & (gh < 16)
        a = tl.load(A + source[:, None] * AS + gh, gm, other=0).to(tl.float32)
        b = tl.load(B + source[:, None] * BS + gh, gm, other=0).to(tl.float32)
        bias = tl.load(BIAS + gh, gh < 16, other=0).to(tl.float32)
        log = tl.load(LOG + gh, gh < 16, other=0).to(tl.float32)
        gate_input = a + bias
        softplus = tl.where(
            gate_input <= 20.0, tl.log(1.0 + tl.exp(gate_input)), gate_input
        )
        tl.store(G + token[:, None] * 16 + gh, -tl.exp(log) * softplus, gm)
        tl.store(BETA + token[:, None] * 16 + gh, tl.sigmoid(b), gm)
