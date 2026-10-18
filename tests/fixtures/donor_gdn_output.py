# SPDX-License-Identifier: Apache-2.0
# Forward adapted from vllm-ascend 9bf964cb (Huawei Technologies, 2025).
"""Owned output destination; allocation remains graph-pool managed."""

import torch


def forward(self, hidden_states, output=None):
    # This adapter deliberately admits only the qualified non-LoRA Qwen35 layout.
    if self.gqa_interleaved_layout or hasattr(self, "in_proj_qkv"):
        raise ValueError("Owned GDN output requires fused Qwen35 QKVZ projections")
    if (getattr(self, "_bs_microflow", False)
            and hidden_states.shape[-1] == 2048
            and hidden_states.dtype == torch.bfloat16
            and getattr(self.in_proj_qkvz, "custom_op", None) is None
            and getattr(self.in_proj_ba, "custom_op", None) is None
            and (self.tp_size, self.key_dim, self.value_dim,
                 self.num_v_heads, self.head_v_dim) == (2, 2048, 4096, 32, 128)
            and getattr(getattr(self.in_proj_qkvz, "weight", None), "dtype", None) == torch.bfloat16
            and getattr(getattr(self.in_proj_ba, "weight", None), "dtype", None) == torch.bfloat16):
        core = hidden_states.new_empty((hidden_states.shape[0], 16, 128))
        z = torch.empty_like(core)
        torch.ops.vllm.betterscale_microflow_gdn(hidden_states, core, z, self.prefix)
        return finish(self, core, z, output)
    mixed, _ = self.in_proj_qkvz(hidden_states)
    tokens = mixed.size(0)
    qkv, z = mixed.split(
        [
            (self.key_dim * 2 + self.value_dim) // self.tp_size,
            self.value_dim // self.tp_size,
        ],
        dim=-1,
    )
    z = z.reshape(tokens, -1, self.head_v_dim)
    ba, _ = self.in_proj_ba(hidden_states)
    b, a = self._split_ba_for_tp(ba)
    if not getattr(self, "_betterscale_strided_gates", False):
        b, a = b.contiguous(), a.contiguous()
    core = torch.empty(
        (tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    # Retain the donor's compiler-visible mutating custom-op boundary.
    torch.ops.vllm.qwen_gdn_attention_core(qkv, b, a, core, self.prefix, False)
    return finish(self, core, z, output)


def finish(self, core, z, output=None):
    tokens = core.shape[0]
    result = self.norm(
        core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim)
    ).reshape(tokens, -1)
    result, _ = self.out_proj(result)
    if output is not None:
        output[:tokens] = result
    return result


def install():
    from vllm_ascend.patch.worker.patch_qwen3_5 import _GDN_PATCH_TARGET
    from .integration import forward_core

    if _GDN_PATCH_TARGET._forward_core is not forward_core:
        raise ValueError("Owned GDN output requires the Qwen35 output writer")
    _GDN_PATCH_TARGET.forward = forward
