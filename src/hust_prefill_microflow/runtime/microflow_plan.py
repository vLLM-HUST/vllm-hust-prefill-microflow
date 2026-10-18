# SPDX-License-Identifier: Apache-2.0
"""Host-only two-slab routing, preserving recurrent 64-token chunk boundaries.

No scheduling decision reads device state. Returned maps are copied into the
existing publication bank; slab capacities and captured tensor shapes stay fixed.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Slab:
    capacity: int
    start: int
    lengths: tuple[int, ...]
    rows: tuple[int, ...]
    continued: tuple[bool, ...]

    @property
    def tokens(self):
        return sum(self.lengths)


def slabs(capacity, lengths, verification):
    if capacity not in (4096, 8192):
        raise ValueError("Microflow has no admitted split for this capture capacity")
    if (not lengths or len(lengths) != len(verification)
            or any(n <= 0 for n in lengths) or sum(lengths) > capacity):
        raise ValueError("Invalid microflow live request lengths")
    half = capacity // 2
    cut, cursor = min(half, sum(lengths)), 0
    for length, verify in zip(lengths, verification):
        if cursor < cut < cursor + length:
            # Never divide a verification row. Prefill pieces must start on the
            # ORIGINAL sequence's chunk grid, not the global token grid.
            cut = cursor if verify else cursor + (cut - cursor) // 64 * 64
            break
        cursor += length
    if sum(lengths) - cut > half + 64:
        raise ValueError("Verification row crosses the supported split envelope")
    result = []
    for start, end, limit in ((0, cut, half), (cut, sum(lengths), half + 64)):
        ls, rows, continued, cursor = [], [], [], 0
        for row, length in enumerate(lengths):
            lo, hi = max(start, cursor), min(end, cursor + length)
            if lo < hi:
                ls.append(hi - lo)
                rows.append(row)
                continued.append(lo != cursor)
            cursor += length
        result.append(Slab(limit, start, tuple(ls), tuple(rows), tuple(continued)))
    return tuple(result)


def field_specs(capacity, requests, width):
    """Flat tensor fields: MTPFrame already publishes top-level core tensors."""
    result = {
        'mf_join': ((capacity,), 'int64'),
        'mf_z': ((capacity,), 'int64'),
        'mf_verify': ((requests * width,), 'int64'),
    }
    for i, size in enumerate((capacity // 2, capacity // 2 + 64)):
        for name, shape, dtype in (
            ('input', (size,), 'int64'), ('pre_map', (size,), 'int64'),
            ('cu', (requests + 2,), 'int32'),
            ('pre_cu', (requests + 2,), 'int64'),
            ('pre_slots', (requests + 1, 1), 'int32'),
            ('ver_slots', (requests + 1, 1), 'int32'),
            ('initial', (requests + 1,), 'bool'),
            ('row_ids', (requests + 1,), 'int64'),
        ):
            result[f'mf_{i}_{name}'] = (shape, dtype)
        for block in (64, 256, 1216):
            result[f'mf_{i}_idx{block}'] = (((size + block - 1) // block + requests - 1, 2), 'int64')
    return result


def fill_fields(fields, capacity, requests, lengths, roles, slots, initial):
    import numpy as np

    plans = slabs(capacity, lengths, roles)
    for field in fields.values():
        field.fill(0)
    raw_starts, pre_starts, ver_starts = [], [], []
    raw = pre = ver = 0
    for length, verify in zip(lengths, roles):
        raw_starts.append(raw); pre_starts.append(pre); ver_starts.append(ver)
        raw += length
        if verify:
            ver += length
        else:
            pre += length
    for i, plan in enumerate(plans):
        def dst(name):
            return fields[f'mf_{i}_{name}']
        base = 0 if i == 0 else plans[0].capacity
        dst('pre_slots').fill(-1); dst('ver_slots').fill(-1)
        local_raw = local_pre = 0
        pre_lengths = []
        dst('input')[:plan.tokens] = np.arange(plan.start, plan.start + plan.tokens)
        for j, (row, length, continuation) in enumerate(zip(plan.rows, plan.lengths, plan.continued)):
            dst('cu')[j + 1] = local_raw + length
            dst('row_ids')[j] = row
            dst('initial')[j] = bool(initial[row]) or continuation
            dst('ver_slots' if roles[row] else 'pre_slots')[j, 0] = slots[row][0]
            global_raw = plan.start + local_raw
            encoded_raw = base + np.arange(local_raw, local_raw + length)
            fields['mf_z'][global_raw:global_raw + length] = encoded_raw
            if roles[row]:
                target = ver_starts[row]
                fields['mf_verify'][target:target + length] = encoded_raw
            else:
                dst('pre_map')[local_pre:local_pre + length] = np.arange(local_raw, local_raw + length)
                target = pre_starts[row] + global_raw - raw_starts[row]
                fields['mf_join'][target:target + length] = base + np.arange(local_pre, local_pre + length)
                local_pre += length
                pre_lengths.append(length)
            local_raw += length
        dst('cu')[len(plan.rows) + 1:] = local_raw
        np.cumsum(pre_lengths, out=dst('pre_cu')[1:len(pre_lengths) + 1])
        dst('pre_cu')[len(pre_lengths) + 1:] = local_pre
        for block in (64, 256, 1216):
            indices = dst(f'idx{block}')
            indices[:, 0] = requests
            cursor = 0
            for row, length in enumerate(pre_lengths):
                count = (length + block - 1) // block
                indices[cursor:cursor + count, 0] = row
                indices[cursor:cursor + count, 1] = np.arange(count)
                cursor += count
    return plans


def allocate_fields(core, device):
    import torch
    for name, (shape, dtype) in field_specs(core.capacity, core.requests, core.width).items():
        setattr(core, name, torch.zeros(shape, dtype=getattr(torch, dtype), device=device))
