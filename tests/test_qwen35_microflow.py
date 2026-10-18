"""Microflow rollback and capture-shape policy, without importing NPU code."""
import sys
import unittest
from unittest.mock import patch
from hust_prefill_microflow.runtime.microflow import FLAG, enabled, install, partition


class Policy(unittest.TestCase):
    def test_explicit_boolean_and_default_off(self):
        self.assertFalse(enabled({}))
        self.assertFalse(enabled({FLAG: '0'}))
        self.assertTrue(enabled({FLAG: '1'}))
        for value in ('true', '2', '', 'yes'):
            with self.assertRaises(ValueError):
                enabled({FLAG: value})

    def test_off_does_not_import_or_install_npu_adapter(self):
        with patch.dict('os.environ', {FLAG: '0'}):
            with patch.dict(sys.modules, {'hust_prefill_microflow.runtime.microflow_attention': None}):
                install()

    def test_capture_shape_not_live_request_count(self):
        for tokens, pieces in ((1024, 2), (2048, 2), (4096, 4), (8192, 4)):
            edges = partition(tokens)
            self.assertEqual(edges, tuple(tokens * i // pieces for i in range(pieces + 1)))
            self.assertTrue(all((b-a) % 64 == 0 for a,b in zip(edges, edges[1:])))
        for tokens in (0, 1, 48, 80, 128, 1023, 2049, 4095, 4097):
            self.assertEqual(partition(tokens), ())


class SlabRouting(unittest.TestCase):
    def test_single_sequence_cut_is_chunk_aligned(self):
        from hust_prefill_microflow.runtime.microflow_plan import slabs
        a,b=slabs(4096,[4096],[False])
        self.assertEqual((a.lengths,b.lengths),((2048,),(2048,)))
        self.assertEqual(b.continued,(True,))

    def test_ragged_cut_is_relative_to_request_not_global_position(self):
        from hust_prefill_microflow.runtime.microflow_plan import slabs
        lengths=[1537,1001,777,511,173,97]
        a,b=slabs(4096,lengths,[False]*6)
        self.assertEqual(b.start,1537+7*64)
        self.assertEqual(a.tokens+b.tokens,sum(lengths))
        self.assertLessEqual(a.tokens,a.capacity)
        self.assertLessEqual(b.tokens,b.capacity)
        self.assertEqual(b.rows,(1,2,3,4,5))
        self.assertEqual(b.continued,(True,False,False,False,False))

    def test_verification_is_never_split_and_empty_second_slab_is_valid(self):
        from hust_prefill_microflow.runtime.microflow_plan import slabs
        a,b=slabs(4096,[2047,3,2046],[False,True,False])
        self.assertEqual(b.start,2047)
        self.assertEqual(b.lengths,(3,2046))
        self.assertFalse(any(b.continued))
        a,b=slabs(4096,[3,17,100],[True,False,False])
        self.assertEqual(b.tokens,0)
        self.assertEqual(a.tokens,120)

class PublishedRouting(unittest.TestCase):
    def test_changing_mixed_lengths_overwrites_all_maps_and_sentinels(self):
        import numpy as np
        from hust_prefill_microflow.runtime.microflow_plan import field_specs, fill_fields
        fields={k:np.empty(shape,dtype=dtype) for k,(shape,dtype) in field_specs(4096,16,3).items()}
        for lengths,roles in (([1537,3,2556],[False,True,False]),([3,17,100],[True,False,False])):
            slots=np.arange(len(lengths)*3).reshape(-1,3)
            plans=fill_fields(fields,4096,16,lengths,roles,slots,[False]*len(lengths))
            raw=np.arange(sum(lengths))
            padded=[];prefilled=[]
            for i,plan in enumerate(plans):
                x=raw[fields[f'mf_{i}_input']]
                padded.extend(x.tolist())
                prefilled.extend(x[fields[f'mf_{i}_pre_map']].tolist())
                ends=fields[f'mf_{i}_pre_cu'];ix=fields[f'mf_{i}_idx64']
                self.assertEqual(ends[-1],ends[-2])
                for row,chunk in ix:
                    self.assertLessEqual(ends[row]+chunk*64,ends[row+1])
            np.testing.assert_array_equal(np.array(padded)[fields['mf_z'][:len(raw)]],raw)
            start=0;pre=[];verify=[]
            for length,role in zip(lengths,roles):
                (verify if role else pre).extend(range(start,start+length));start+=length
            np.testing.assert_array_equal(np.array(prefilled)[fields['mf_join'][:len(pre)]],pre)
            np.testing.assert_array_equal(np.array(padded)[fields['mf_verify'][:len(verify)]],verify)


@unittest.skipUnless(__import__('importlib').util.find_spec('torch'), 'CPU Torch required for geometry checks')
class AttentionAdmission(unittest.TestCase):
    def test_actual_ascend_subclass_and_unknown_projection_fallback(self):
        import ast
        from pathlib import Path
        from types import SimpleNamespace as S
        import torch
        path = Path(__file__).parents[1] / 'src/hust_prefill_microflow/runtime/microflow_attention.py'
        node = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'admitted')
        namespace = {'torch': torch, 'partition': partition}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        hidden = torch.empty((4096, 2048), device='meta', dtype=torch.bfloat16)
        attn = S(num_heads=8, num_kv_heads=1, head_dim=256, attn_output_gate=True,
                 rotary_emb=S(rotary_dim=64, mrope_interleaved=True,
                              mrope_section=(11, 11, 10)))
        layer = S(layer_type='full_attention', layer_scale=False, self_attn=attn)
        for name, expected in (('QKVParallelLinear', True),
                               ('AscendQKVParallelLinear', True), ('UnknownLinear', False)):
            attn.qkv_proj = type(name, (), {})()
            attn.qkv_proj.weight = S(dtype=torch.bfloat16)
            self.assertEqual(namespace['admitted'](layer, hidden), expected)
        attn.qkv_proj = type('AscendQKVParallelLinear', (), {})()
        attn.qkv_proj.weight = S(dtype=torch.bfloat16)
        layer._bs_microflow_key = 'mtp.layers.0'
        self.assertFalse(namespace['admitted'](layer, hidden))
        layer._bs_microflow_key = 'language_model.model.layers.3'
        attn.qkv_proj.custom_op = object()
        self.assertFalse(namespace['admitted'](layer, hidden))
        attn.qkv_proj.custom_op = None
        del attn.qkv_proj.weight
        self.assertFalse(namespace['admitted'](layer, hidden))


if __name__ == '__main__':
    unittest.main()
