# SPDX-License-Identifier: Apache-2.0
"""Two fixed-capacity slabs, dynamic routing, joined recurrent H/O barrier.

Projection→prep→KKT→solve→WU keeps the measured frontload schedule. Publication
may change request lengths/roles without changing captured allocations. Conv
pieces crossing a request are serialized; recurrent state is updated only by
the original joined H/O and verification consumers.
"""
import torch

from .microflow_layout import join_prefill, join_tokens
from .microflow_native import Native
from .microflow_streams import get as get_streams
from .microflow_preprocess import preprocess_kernel
from .microflow_triangle import solve_tril_16x16_kernel
from betterscale.models.qwen35.mixed_layout import cumulative_kernel, restore_token


class Pipeline:
    def __init__(self, core, device):
        self.core = core
        self.sizes = (core.capacity // 2, core.capacity // 2 + 64)
        self.native = Native()
        self.streams = get_streams(torch.device(device))
        self.mask = torch.ones(64, 64, dtype=torch.float32, device=device).tril(-1)
        # Invocation is joined; both jobs may reuse native scratch serially.
        self.kkt_scratch = torch.empty(24 * 2 * 4096, dtype=torch.float32, device=device)
        self.wu_scratch = torch.empty(24 * 2 * 12288, dtype=torch.float32, device=device)

    def field(self, i, name):
        # MTPFrame replaces these with published-bank views after construction.
        return getattr(self.core, f'mf_{i}_{name}')

    def jobs(self, x):
        jobs = []
        for i, size in enumerate(self.sizes):
            j = {'T': size, 'x': x[:size] if i == 0 else x.index_select(0, self.field(i, 'input'))}
            for name, shape, dtype in (
                ('conv',(size,4096),x.dtype),
                ('q',(1,8,size,128),x.dtype), ('k',(1,size,8,128),x.dtype),
                ('v',(1,size,16,128),x.dtype), ('g',(1,size,16),torch.float32),
                ('beta',(1,size,16),x.dtype), ('gh',(1,16,size),torch.float32),
                ('bh',(1,16,size),x.dtype), ('matrix',(1,size,16,64),torch.float32),
                ('ad',(1,size,16,16),torch.float32), ('ai',(1,size,16,64),x.dtype),
                ('w',(1,16,size,128),x.dtype), ('u',(1,16,size,128),x.dtype),
            ):
                j[name] = torch.empty(shape, dtype=dtype, device=x.device)
            j['accepted'] = self.core.accepted.index_select(0, self.field(i, 'row_ids'))
            jobs.append(j)
        return jobs

    def run(self, layer, x, output, z):
        from vllm_ascend.ops.triton.fla.solve_tril import merge_16x16_to_64x64_inverse_kernel
        from betterscale.models.qwen35.preprocess import preprocess
        from betterscale.models.qwen35.decode_kv import fused_recurrent_gated_delta_rule_fwd

        meta, streams = self.core, self.streams
        jobs = self.jobs(x)
        conv_state, state = layer.kv_cache
        # Native conv requires this layout; materialize it once, not once per
        # role/slab (especially not on the contended 24-AIV queue).
        weight = layer.conv1d.weight.view(layer.conv1d.weight.size(0), 4).T.contiguous()

        def project(i):
            j = jobs[i]
            j['mixed'], _ = layer.in_proj_qkvz(j['x'])
            j['ba'], _ = layer.in_proj_ba(j['x'])
            j['packed'] = j['mixed'][:, :4096].contiguous()

        def prepare(i):
            j = jobs[i]
            for verify in (False, True):
                torch.ops._C_ascend.npu_causal_conv1d_custom(
                    j['conv'], j['packed'], weight, conv_state=conv_state,
                    bias_opt=None, query_start_loc_opt=self.field(i, 'cu'),
                    cache_indices_opt=self.field(i, 'ver_slots' if verify else 'pre_slots'),
                    initial_state_mode_opt=None if verify else self.field(i, 'initial'),
                    num_accepted_tokens_opt=j['accepted'] if verify else None,
                    activation_mode=1, pad_slot_id=-1, run_mode=1 if verify else 0,
                )
            preprocess_kernel[(24,)](
                j['conv'], j['ba'][:,16:], j['ba'][:,:16], layer.A_log, layer.dt_bias,
                j['q'],j['k'],j['v'],j['g'],j['beta'],self.field(i,'pre_map'),
                True,True,j['T'],4096,32,32,4,24,num_warps=4,
            )
            cumulative_kernel[(len(self.field(i,'idx256')),)](
                j['g'],j['beta'],j['gh'],j['bh'],self.field(i,'pre_cu'),
                self.field(i,'idx256'),j['T'],num_warps=8,
            )

        def kkt(i):
            j = jobs[i]
            args = [j['k'],j['bh'],j['gh'],self.mask,j['matrix'],self.kkt_scratch]
            self.native.check(self.native.kkt(
                24,torch.npu.current_stream().npu_stream,*[v.data_ptr() for v in args],
                j['T'],self.field(i,'pre_cu').data_ptr(),self.field(i,'idx64').data_ptr(),
                len(self.field(i,'idx64')),
            ))

        def triangle(i):
            j=jobs[i]
            solve_tril_16x16_kernel[(48,)](
                j['matrix'],j['ad'],self.field(i,'pre_cu'),self.field(i,'idx1216'),
                j['T'],16,BT=64,LARGE_BLOCK_T=1216,EXTRACT_SLICE_STRIDE_1=38,
                CORES=48,TASKS=len(self.field(i,'idx1216')),num_warps=1,num_stages=4,
            )

        def merge(i):
            j=jobs[i];cu,indices=self.field(i,'pre_cu'),self.field(i,'idx64')
            merge_16x16_to_64x64_inverse_kernel[(len(indices),16)](
                j['matrix'],j['ad'],j['ai'],cu,indices,j['T'],16,BT=64,num_warps=4,num_stages=3,
            )
            args=[j[n] for n in ('k','v','bh','gh','ai','w','u')]+[self.wu_scratch,cu,indices]
            self.native.check(self.native.wu(
                24,torch.npu.current_stream().npu_stream,*[v.data_ptr() for v in args],
                j['T'],len(indices),
            ))

        full, half = streams.cube, streams.half
        stages = (
            ((full,project,0),), ((full,project,1),(half,prepare,0)),
            ((full,kkt,0),), ((full,triangle,0),),
            ((full,merge,0),(half,prepare,1)),
            ((full,kkt,1),), ((full,triangle,1),), ((full,merge,1),),
        )
        main = torch.npu.current_stream()
        for stage in stages:
            for stream, fn, i in stage:
                stream.wait_stream(main)
                with streams.use(stream):
                    fn(i)
            for stream, _, _ in stage:
                main.wait_stream(stream)
        combined = {}
        for name, heads, dim in (('q',8,128),('k',8,128),('w',16,128),
                                 ('u',16,128),('gh',16,1)):
            dtype = torch.float32 if name == 'gh' else x.dtype
            combined[name] = torch.empty((1,heads,meta.capacity,dim),
                                         dtype=dtype,device=x.device)
        join_prefill[(meta.capacity,)](
            *[j[n] for j in jobs for n in ('q','k','w','u','gh','mixed')],
            meta.mf_join,meta.mf_z,*[combined[n] for n in ('q','k','w','u','gh')],
            z,meta.prefill.cu,meta.cu,meta.cu.numel()-1,*self.sizes,meta.capacity,
        )
        ph = meta.prefill.engine.pool_forward(
            combined['q'],combined['k'],combined['w'],combined['u'],
            combined['gh'].squeeze(-1),state,meta.prefill.cu,meta.prefill.state,
            meta.prefill.indices[64],
        ).transpose(1,2)
        verify_tokens = meta.requests * meta.width
        conv = x.new_empty((verify_tokens,4096));ba = x.new_empty((verify_tokens,32))
        for name,dest,width,stride in (('conv',conv,4096,4096),('ba',ba,32,32)):
            join_tokens[(verify_tokens,)](
                jobs[0][name],jobs[1][name],meta.mf_verify,dest,
                meta.verify.cu,meta.verify.cu.numel()-1,self.sizes[0],stride,0,width,width,
            )
        q,k,v,g,beta=preprocess(conv,ba[:,16:],ba[:,:16],layer.A_log,layer.dt_bias)
        vh,_=fused_recurrent_gated_delta_rule_fwd(
            q,k,v,g,beta,128**-0.5,state,cu_seqlens=meta.verify.cu,
            ssm_state_indices=meta.verify.slots,num_accepted_tokens=meta.verify.accepted,
        )
        restore_token[(meta.capacity,)](
            ph,vh,meta.restore,output,meta.capacity,meta.cu,meta.cu.numel()-1,
        )
        # Projection buffers were created on the full helper stream and consumed
        # by prep/join. Preserve allocator ownership across asynchronous readers.
        for j in jobs:
            for name in ('mixed','ba','packed'):
                j[name].record_stream(half)
                j[name].record_stream(main)
        return output,z


def _gdn_op(x: torch.Tensor, output: torch.Tensor, z: torch.Tensor, prefix: str) -> None:
    from vllm.forward_context import get_forward_context

    ctx = get_forward_context()
    layer = ctx.no_compile_layers[prefix]
    if ctx.attn_metadata is None:
        mixed, _ = layer.in_proj_qkvz(x)
        output.zero_()
        z.copy_(mixed[:,4096:].reshape(-1,16,128))
        return
    meta = ctx.attn_metadata[prefix].owned
    # Dynamo sees one shape-polymorphic mutating op. The captured concrete
    # shape chooses the schedule here, not in the model's dynamic FX graph.
    if x.shape[0] == 4096:
        if meta.microflow is None:
            raise RuntimeError("Microflow GDN publication bank was not initialized")
        meta.microflow.run(layer, x, output, z)
        return
    mixed, _ = layer.in_proj_qkvz(x)
    qkv, gate = mixed.split([4096, 2048], dim=-1)
    ba, _ = layer.in_proj_ba(x)
    b, a = layer._split_ba_for_tp(ba)
    if not getattr(layer, "_betterscale_strided_gates", False):
        b, a = b.contiguous(), a.contiguous()
    torch.ops.vllm.qwen_gdn_attention_core(qkv, b, a, output, prefix, False)
    z.copy_(gate.reshape(-1, 16, 128))


def _gdn_fake(x, output, z, prefix):
    return None


def install():
    from vllm.utils.torch_utils import direct_register_custom_op
    from vllm_ascend.patch.worker.patch_qwen3_5 import _GDN_PATCH_TARGET
    from betterscale.models.qwen35.gdn_output import forward

    if _GDN_PATCH_TARGET.forward is not forward:
        raise ValueError("Microflow requires the owned GDN output writer")
    if getattr(_GDN_PATCH_TARGET, '_bs_microflow', False):
        return
    direct_register_custom_op(op_name='betterscale_microflow_gdn',op_func=_gdn_op,
                              mutates_args=['output','z'],fake_impl=_gdn_fake)
    _GDN_PATCH_TARGET._bs_microflow = True
