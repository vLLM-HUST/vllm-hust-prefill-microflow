"""CPU Dynamo boundary gate: serving marks the token dimension dynamic."""
import ast
from pathlib import Path
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).parents[1] / 'src/hust_prefill_microflow/runtime'


def load_function(file, name, namespace):
    node = next(n for n in ast.parse(((Path(__file__).parent / 'fixtures/donor_gdn_output.py') if file == 'gdn_output.py' else (ROOT / file)).read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), file, 'exec'), namespace)
    return namespace[name]


@unittest.skipIf(torch is None, 'CPU Torch required for Dynamo boundary checks')
class DynamicTokenBoundary(unittest.TestCase):
    def check_shapes(self, fn):
        graphs = []
        def backend(graph, inputs):
            graphs.append(graph)
            return graph.forward
        compiled = torch.compile(fn, backend=backend, fullgraph=True, dynamic=True)
        for tokens in (4096, 48, 128, 1024, 2048, 8192):
            x = torch.zeros(tokens, 2048, dtype=torch.bfloat16)
            torch._dynamo.mark_dynamic(x, 0)
            result = compiled(x)
            self.assertEqual(result.shape[0], tokens)
        self.assertEqual(len(graphs), 1)

    def test_gdn_dispatch_does_not_specialize_token_dimension(self):
        @torch.library.custom_op('bs_microflow_test::gdn', mutates_args=('output', 'z'))
        def op(x: torch.Tensor, output: torch.Tensor, z: torch.Tensor, prefix: str) -> None:
            output.zero_(); z.zero_()
        @op.register_fake
        def fake(x, output, z, prefix):
            return None
        namespace = {'torch': torch, 'finish': lambda layer, core, z, output: core}
        forward = load_function('gdn_output.py', 'forward', namespace)
        layer = S(gqa_interleaved_layout=False, _bs_microflow=True,
                  tp_size=2, key_dim=2048, value_dim=4096, num_v_heads=32,
                  head_v_dim=128, prefix='layer',
                  in_proj_qkvz=S(weight=S(dtype=torch.bfloat16)),
                  in_proj_ba=S(weight=S(dtype=torch.bfloat16)))
        with patch.object(torch.ops.vllm, 'betterscale_microflow_gdn', op, create=True):
            self.check_shapes(lambda x: forward(layer, x))

    def test_attention_dispatch_does_not_specialize_token_dimension(self):
        @torch.library.custom_op('bs_microflow_test::fa', mutates_args=())
        def op(x: torch.Tensor, residual: torch.Tensor | None, cos: torch.Tensor,
               key: str) -> list[torch.Tensor]:
            return [x.new_zeros((x.shape[0], n)) for n in (2048,256,256,2048,2048)]
        @op.register_fake
        def fake(x, residual, cos, key):
            return [x.new_empty((x.shape[0], n)) for n in (2048,256,256,2048,2048)]
        namespace = {'torch': torch, 'selected_rows':
                     lambda attn, positions, x: x.new_empty((3,x.shape[0],64))}
        admitted = load_function('microflow_attention.py', 'admitted', namespace)
        forward = load_function('microflow_attention.py', 'layer_forward', namespace)
        linear = type('AscendQKVParallelLinear', (), {})()
        linear.weight = S(dtype=torch.bfloat16)
        attn = S(qkv_proj=linear, num_heads=8, num_kv_heads=1, head_dim=256,
                 attn_output_gate=True, rotary_emb=S(rotary_dim=64,
                 mrope_interleaved=True,mrope_section=(11,11,10)),
                 attn=lambda q,k,v:q, o_proj=lambda x:(x,None))
        layer = S(layer_type='full_attention', layer_scale=False, self_attn=attn,
                  _bs_microflow_key='layer', post_attention_layernorm=lambda x,r:(x,r),
                  mlp=lambda x:x)
        def run(x):
            if not admitted(layer,x):
                raise AssertionError('qualified geometry unexpectedly rejected')
            return forward(layer,x,None,None)[0]
        with patch.object(torch.ops.vllm, 'betterscale_microflow_fa_prefix', op, create=True):
            self.check_shapes(run)


if __name__ == '__main__':
    unittest.main()
