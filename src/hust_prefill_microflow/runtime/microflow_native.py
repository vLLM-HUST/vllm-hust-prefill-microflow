# SPDX-License-Identifier: Apache-2.0
"""Launch the fixed 24/24 KKT and 24/48 WU binaries; no runtime compilation."""
import ctypes as C
import os
from pathlib import Path


def directory(env=None):
    env = os.environ if env is None else env
    value = env.get('BETTERSCALE_MICROFLOW_NATIVE_DIR')
    if not value:
        raise ValueError('Set BETTERSCALE_MICROFLOW_NATIVE_DIR to separately built KKT/WU libraries')
    return Path(value)


class Native:
    def __init__(self):
        root = directory()
        self.kkt_lib = C.CDLL(str(root / 'libbs_microflow_kkt.so'))
        self.wu_lib = C.CDLL(str(root / 'libbs_microflow_wu.so'))
        self.kkt = self.kkt_lib.aclrtlaunch_kkt11_varlen
        self.kkt.argtypes = [C.c_uint32,C.c_void_p] + [C.c_void_p]*6 + [C.c_int,C.c_void_p,C.c_void_p,C.c_int]
        self.kkt.restype = C.c_uint32
        self.wu = self.wu_lib.aclrtlaunch_wu2
        self.wu.argtypes = [C.c_uint32,C.c_void_p] + [C.c_void_p]*10 + [C.c_int]*2
        self.wu.restype = C.c_uint32

    @staticmethod
    def check(status):
        if status:
            raise RuntimeError(f'Microflow native launch failed: {status}')
