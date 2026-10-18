# SPDX-License-Identifier: Apache-2.0
"""Two-slab joins. Scalar token routing keeps each channel copy contiguous."""
from vllm.triton_utils import tl, triton


@triton.jit
def join_tokens(A, B, MAP, OUT, CU, LAST: tl.constexpr,
                S0: tl.constexpr, STRIDE: tl.constexpr,
                OFFSET: tl.constexpr, WIDTH: tl.constexpr,
                BLOCK: tl.constexpr):
    token = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    live = token < tl.load(CU + LAST)
    source = tl.load(MAP + token)
    if source < S0:
        value = tl.load(A + source * STRIDE + OFFSET + col, col < WIDTH, 0)
    else:
        value = tl.load(B + (source - S0) * STRIDE + OFFSET + col, col < WIDTH, 0)
    tl.store(OUT + token * WIDTH + col, tl.where(live, value, 0), col < WIDTH)


@triton.jit
def join_prefill(Q0,K0,W0,U0,G0,Z0,Q1,K1,W1,U1,G1,Z1,
                 MAP,ZMAP,Q,K,W,U,G,Z,CU,RAW_CU,LAST:tl.constexpr,
                 S0:tl.constexpr,S1:tl.constexpr,T:tl.constexpr):
    """One routing pass for all H/O inputs and the raw-order output gate."""
    token=tl.program_id(0)
    h=tl.arange(0,16)[:,None];d=tl.arange(0,128)[None,:]
    live=token<tl.load(CU+LAST)
    source=tl.load(MAP+token)
    if source<S0:
        q=tl.load(Q0+(h*S0+source)*128+d,h<8,0)
        k=tl.load(K0+(source*8+h)*128+d,h<8,0)
        w=tl.load(W0+(h*S0+source)*128+d)
        u=tl.load(U0+(h*S0+source)*128+d)
        g=tl.load(G0+h*S0+source)
    else:
        source-=S0
        q=tl.load(Q1+(h*S1+source)*128+d,h<8,0)
        k=tl.load(K1+(source*8+h)*128+d,h<8,0)
        w=tl.load(W1+(h*S1+source)*128+d)
        u=tl.load(U1+(h*S1+source)*128+d)
        g=tl.load(G1+h*S1+source)
    off=(h*T+token)*128+d
    tl.store(Q+off,tl.where(live,q,0),h<8)
    tl.store(K+off,tl.where(live,k,0),h<8)
    tl.store(W+off,tl.where(live,w,0))
    tl.store(U+off,tl.where(live,u,0))
    tl.store(G+h*T+token,tl.where(live,g,0))
    raw_live=token<tl.load(RAW_CU+LAST)
    raw=tl.load(ZMAP+token)
    if raw<S0:
        z=tl.load(Z0+raw*6144+4096+h*128+d)
    else:
        z=tl.load(Z1+(raw-S0)*6144+4096+h*128+d)
    tl.store(Z+token*2048+h*128+d,tl.where(raw_live,z,0))
