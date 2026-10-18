# SPDX-License-Identifier: Apache-2.0
"""Joined per-device execution streams; no shared activation buffers."""
import ctypes
from contextlib import contextmanager

import torch


class Streams:
    def __init__(self):
        self.acl = ctypes.CDLL("libascendcl.so")
        for name in ("aclrtUseStreamResInCurrentThread", "aclrtUnuseStreamResInCurrentThread"):
            getattr(self.acl, name).argtypes = [ctypes.c_void_p]
            getattr(self.acl, name).restype = ctypes.c_int
        self.acl.aclrtSetStreamResLimit.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32]
        self.acl.aclrtSetStreamResLimit.restype = ctypes.c_int
        self.cube = torch.npu.Stream()
        self.vector = torch.npu.Stream()
        self.half = torch.npu.Stream()
        for stream, quota in ((self.cube, 48), (self.vector, 48), (self.half, 24)):
            self.check(self.acl.aclrtSetStreamResLimit(stream.npu_stream, 1, quota))

    @staticmethod
    def check(status):
        if status:
            raise RuntimeError(f"Microflow stream resource operation failed: {status}")

    @contextmanager
    def use(self, stream):
        self.check(self.acl.aclrtUseStreamResInCurrentThread(stream.npu_stream))
        try:
            with torch.npu.stream(stream):
                yield
        finally:
            self.check(self.acl.aclrtUnuseStreamResInCurrentThread(stream.npu_stream))

    @contextmanager
    def joined(self):
        main = torch.npu.current_stream()
        self.cube.wait_stream(main)
        self.vector.wait_stream(main)
        self.half.wait_stream(main)
        try:
            yield self
        finally:
            main.wait_stream(self.cube)
            main.wait_stream(self.vector)
            main.wait_stream(self.half)


_instances = {}

def get(device):
    device = torch.device(device)
    if device.index is None:
        device = torch.device(device.type, torch.npu.current_device())
    if device not in _instances:
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("Microflow requires eager stream warmup before capture")
        _instances[device] = Streams()
    return _instances[device]
