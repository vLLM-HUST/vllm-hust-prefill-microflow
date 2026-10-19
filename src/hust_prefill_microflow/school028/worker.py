"""Explicit worker adapter, preserving the pinned BetterScale lifecycle."""

import os

from betterscale.qwen35_worker import Worker as HostWorker

from . import FLAG, event, preflight


class Worker(HostWorker):
    def load_model(self, *args, **kwargs):
        if os.environ.get(FLAG) != "1":
            raise ValueError("Select school FA through Extension Manager")
        preflight()
        import torch
        from hust_prefill_microflow.runtime import school028_attention as attention
        from hust_prefill_microflow.runtime.microflow_streams import get

        active = os.environ["HUST_MICROFLOW_ARM"] == "on"
        if active:
            get(torch.device("npu", torch.npu.current_device()))
            attention.install()
        result = super().load_model(*args, **kwargs)
        if active:
            self._qualify_prefix(attention)
        event("model_loaded", rank=torch.distributed.get_rank(), active=active,
              gdn_modified=False, draft_modified=False,
              adapter=attention.__file__)
        self._microflow_active = active
        self._install_graph_receipts()
        return result

    def _qualify_prefix(self, attention):
        from contextlib import nullcontext
        from types import SimpleNamespace
        import torch

        target = [layer for key, layer in attention._layers.items()
                  if not key.startswith("mtp.") and layer.layer_type == "full_attention"]
        if len(target) != 10:
            raise ValueError("Expected exactly ten target FA layers")
        layer = target[0]
        main = torch.npu.current_stream()
        serial = SimpleNamespace(cube=main, vector=main, joined=nullcontext,
                                 use=torch.npu.stream)
        with torch.inference_mode():
            for tokens in (1024, 2048, 4096):
                x = torch.sin(torch.arange(tokens * 2048, device="npu", dtype=torch.float32)
                              .reshape(tokens, 2048) / 17).to(torch.bfloat16)
                residual = x * 0.25
                positions = torch.arange(tokens, device="npu").expand(3, -1)
                cos = attention.selected_rows(layer.self_attn, positions, x)
                if not attention.admitted(layer, x):
                    raise ValueError("Target FA layer failed shape admission")
                expected = attention.prefix(layer, x, residual, cos, streams=serial)
                actual = attention.prefix(layer, x, residual, cos)
                torch.npu.synchronize()
                exact = [torch.equal(a, b) for a, b in zip(actual, expected)]
                event("partition_equivalence", tokens=tokens, exact=exact,
                      rank=torch.distributed.get_rank())
                if not all(exact):
                    raise ValueError("Parallel/serial same-partition output mismatch")

    def _install_graph_receipts(self):
        from functools import wraps
        from vllm.forward_context import get_forward_context
        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

        original = ACLGraphWrapper.__call__
        if getattr(original, "_microflow_receipts", False):
            return

        @wraps(original)
        def call(wrapper, *args, **kwargs):
            ctx = get_forward_context()
            descriptor = ctx.batch_descriptor
            entry = wrapper.concrete_aclgraph_entries.get(descriptor)
            replay = (entry is not None and entry.aclgraph is not None
                      and ctx.cudagraph_runtime_mode == wrapper.runtime_mode)
            result = original(wrapper, *args, **kwargs)
            if replay:
                event("graph_replay", tokens=descriptor.num_tokens,
                      draft=wrapper.use_eagle,
                      active=os.environ["HUST_MICROFLOW_ARM"] == "on")
            return result

        call._microflow_receipts = True
        ACLGraphWrapper.__call__ = call
