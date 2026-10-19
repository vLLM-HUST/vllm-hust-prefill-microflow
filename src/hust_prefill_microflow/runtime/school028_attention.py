# SPDX-License-Identifier: Apache-2.0
# Layer tail adapted from vllm-ascend 9bf964cb (Huawei Technologies, 2025).
"""Full-Vector FA prefix scheduling; attention/State writes remain donor-owned.

Unlike the standalone prefix probe, KV scatter stays inside the original
attention call after the joined prefix. No duplicate cache write is introduced.
"""
from functools import wraps
from weakref import WeakValueDictionary

import torch

from .microflow import partition


def selected_rows(attn, positions, hidden):
    # Preserve the school host's RoPE lookup; no donor rope-reuse dependency.
    return attn.rotary_emb.cos_sin_cache[positions].to(
        device=hidden.device, dtype=hidden.dtype
    )

_layers = WeakValueDictionary()


def admitted(layer, hidden):
    # MTP is a separate decode-only model, not a target prefill window.
    # Preserve its compiler-visible operator sequence, including small graphs.
    if getattr(layer, "_bs_microflow_key", "").startswith("mtp."):
        return False
    if (layer.layer_type != "full_attention" or layer.layer_scale
            or layer.use_attn_reduce_scatter_for_moe):
        return False
    attn = layer.self_attn
    return (
        hidden.dtype == torch.bfloat16
        and hidden.shape[-1] == 2048
        and (attn.num_heads, attn.num_kv_heads, attn.head_dim) == (8, 1, 256)
        and getattr(attn.qkv_proj, "custom_op", None) is None
        and attn.attn_output_gate
        and attn.rotary_emb.rotary_dim == 64
        and attn.rotary_emb.mrope_interleaved
        and tuple(attn.rotary_emb.mrope_section) == (11, 11, 10)
        and type(attn.qkv_proj).__name__ in ("QKVParallelLinear", "AscendQKVParallelLinear")
        and getattr(getattr(attn.qkv_proj, "weight", None), "dtype", None) == torch.bfloat16
    )


def prefix(layer, x, residual, cos, *, streams=None):
    from .microflow_rope import split_qkv_rmsnorm_mrope_kernel as post
    from .microflow_streams import get as get_streams

    attn, tokens = layer.self_attn, x.shape[0]
    edges = partition(tokens)
    if not edges:
        # Keep token-count specialization inside the opaque custom op, so the
        # surrounding vLLM graph can retain its marked dynamic token dimension.
        if residual is None:
            norm = layer.input_layernorm(x)
            next_residual = x.clone()  # Functional op outputs must not alias x.
        else:
            norm, next_residual = layer.input_layernorm(x, residual)
        qkv, _ = attn.qkv_proj(norm)
        outputs = torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(
            qkv=qkv,q_weight=1.0+attn.q_norm.weight,k_weight=1.0+attn.k_norm.weight,
            cos_sin=cos,num_q_heads=8,num_kv_heads=1,head_size=256,
            eps=attn.config.rms_norm_eps,mrope_section=[11,11,10],
            is_interleaved=True,rope_dim=64,has_gate=True,
        )
        return (*outputs,next_residual)
    streams = streams or get_streams(x.device)
    qw, kw = 1.0 + attn.q_norm.weight, 1.0 + attn.k_norm.weight
    outputs = [x.new_empty((tokens, n)) for n in (2048, 256, 256, 2048)]
    residual_out = torch.empty_like(x)
    projections, norm_outputs, events = [], [], []
    with streams.joined():
        # All input Norms are frontloaded on one Vector queue. GEMMs begin
        # immediately after their own Norm, not after all Norms finish.
        for lo, hi in zip(edges, edges[1:]):
            with streams.use(streams.vector):
                if residual is None:
                    norm = layer.input_layernorm(x[lo:hi])
                    next_residual = x[lo:hi]
                else:
                    norm, next_residual = layer.input_layernorm(x[lo:hi], residual[lo:hi])
                norm_outputs.append(norm)
                residual_out[lo:hi].copy_(next_residual)
                ready = torch.npu.Event()
                ready.record(streams.vector)
            streams.cube.wait_event(ready)
            with streams.use(streams.cube):
                projected, _ = attn.qkv_proj(norm)
                projections.append(projected)
                done = torch.npu.Event()
                done.record(streams.cube)
            events.extend((ready, done))
        for i, (lo, hi) in enumerate(zip(edges, edges[1:])):
            streams.vector.wait_event(events[2 * i + 1])
            with streams.use(streams.vector):
                n = hi - lo
                post[(min(n, 48),)](
                    projections[i], qw, None, kw, None, cos[:, lo:hi],
                    *[out[lo:hi] for out in outputs], n, tokens,
                    n % 48 or 48, (n + 47) // 48, n // 48,
                    8, 1, 256, 2048, 256, attn.config.rms_norm_eps,
                    11, 11, 10, False, True, 64, 32, True, 2048,
                )
    # Allocations made on helper streams are consumed on the caller stream.
    # Keep intermediates alive through submission; record the cross-stream reads.
    for tensor in projections + norm_outputs:
        tensor.record_stream(streams.vector)
        tensor.record_stream(streams.cube)
    return (*outputs, residual_out)


def _prefix_op(x: torch.Tensor, residual: torch.Tensor | None,
               cos: torch.Tensor, key: str) -> list[torch.Tensor]:
    from hust_prefill_microflow.school028 import event
    event("prefix", layer=key, tokens=x.shape[0], split=bool(partition(x.shape[0])),
          capturing=torch.npu.is_current_stream_capturing())
    return list(prefix(_layers[key], x, residual, cos))


def _prefix_fake(x, residual, cos, key):
    return [x.new_empty((x.shape[0], n)) for n in (2048, 256, 256, 2048, 2048)]


def layer_forward(layer, hidden_states, residual, positions):
    # Keep shared rotary selection visible to the outer model graph.
    cos = selected_rows(layer.self_attn, positions, hidden_states)
    q, k, v, gate, residual = torch.ops.vllm.betterscale_microflow_fa_prefix(
        hidden_states, residual, cos, layer._bs_microflow_key
    )
    attn = layer.self_attn
    hidden_states = attn.attn(q, k, v)
    hidden_states = hidden_states * torch.sigmoid(gate)
    hidden_states, _ = attn.o_proj(hidden_states)
    hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)
    return layer.mlp(hidden_states), residual


def install():
    from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer as Layer
    from vllm.model_executor.models.qwen3_next import Qwen3NextDecoderLayer
    from vllm.utils.torch_utils import direct_register_custom_op

    if getattr(Layer, "_bs_microflow_installed", False):
        return
    if Layer.forward is not Qwen3NextDecoderLayer.forward:
        raise ValueError("School FA requires the pinned inherited decoder forward")
    direct_register_custom_op(
        op_name="betterscale_microflow_fa_prefix", op_func=_prefix_op,
        mutates_args=[], fake_impl=_prefix_fake,
    )
    original_init, original_forward = Layer.__init__, Layer.forward

    @wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        key = kwargs.get("prefix", args[2] if len(args) > 2 else "")
        if not key or key in _layers:
            raise ValueError("Microflow requires unique decoder-layer prefixes")
        self._bs_microflow_key = key
        _layers[key] = self

    @wraps(original_forward)
    def forward(self, hidden_states, residual, positions=None, **kwargs):
        if not admitted(self, hidden_states):
            return original_forward(self, hidden_states, residual, positions, **kwargs)
        return layer_forward(self, hidden_states, residual, positions)

    Layer.__init__, Layer.forward = init, forward
    Layer._bs_microflow_installed = True
