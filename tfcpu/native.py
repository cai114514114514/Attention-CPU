"""NativeCPU: run the v2 Transformer CPU with a small C engine (tfcpu/csrc/engine.c).

Same weights, same operations as FastCPU, minus the Python/PyTorch dispatch
overhead: one call runs many steps. The C source is compiled on first use with
the system C compiler (`cc`, or $CC) into tfcpu/_build/.
"""
from __future__ import annotations

import ctypes as C
import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np
import torch

from .fast import FastCPU
from .isa import NMEM, NOPS, NREG, Program
from .nn import LoopedTransformer
from .reference import State, initial_state
from .routing import NCOMBO, route
from .state import CPU, MEM0, NTOK, decode, encode

_SRC = Path(__file__).with_name("csrc") / "engine.c"
_BUILD = Path(__file__).with_name("_build")
_lib = None

f32p, i32p = C.POINTER(C.c_float), C.POINTER(C.c_int32)


class _CSR(C.Structure):
    _fields_ = [("rows", C.c_int32), ("cols", C.c_int32), ("indptr", i32p), ("indices", i32p), ("data", f32p)]


class _Model(C.Structure):
    _fields_ = [
        ("T", C.c_int32), ("d", C.c_int32), ("cpu", C.c_int32), ("mem0", C.c_int32), ("nmem", C.c_int32), ("halted", C.c_int32),
        ("out", C.c_int32), ("out_flag", C.c_int32),
        ("dk1", C.c_int32), ("dv1", C.c_int32), ("q1", _CSR), ("o1", _CSR), ("K1", f32p), ("V1", f32p),
        ("m1a", _CSR), ("m1b", _CSR), ("c1a", f32p), ("c1b", f32p),
        ("H", C.c_int32), ("R", C.c_int32), ("dk2", C.c_int32), ("dv2", C.c_int32),
        ("q2", _CSR), ("k2", _CSR), ("v2", _CSR), ("o2", _CSR),
        ("K2", f32p), ("V2", f32p), ("Qs", f32p), ("best", f32p),
        ("m2a", _CSR), ("m2b", _CSR), ("c2a", f32p), ("c2b", f32p),
        ("m1bT", _CSR), ("m2bT", _CSR),
        ("f_op0", C.c_int32), ("f_rd0", C.c_int32), ("f_ra0", C.c_int32), ("f_rb0", C.c_int32),
        ("nops", C.c_int32), ("nreg", C.c_int32),
        ("r1_off", i32p), ("r1_idx", i32p), ("r2_off", i32p), ("r2_idx", i32p),
        ("rm_idx", i32p), ("rm_n", C.c_int32),
    ]


def _load_lib():
    global _lib
    if _lib is None:
        src = _SRC.read_bytes()
        so = _BUILD / f"engine-{hashlib.sha256(src).hexdigest()[:12]}.so"
        if not so.exists():
            _BUILD.mkdir(exist_ok=True)
            cc = os.environ.get("CC", "cc")
            tmp = so.with_suffix(".tmp.so")
            subprocess.run([cc, "-O3", "-std=c11", "-shared", "-fPIC", "-o", str(tmp), str(_SRC)], check=True)
            tmp.replace(so)
        _lib = C.CDLL(str(so))
        _lib.tf_run.argtypes = [C.POINTER(_Model), f32p, C.c_int64, f32p, C.c_int64, C.POINTER(C.c_int64)]
        _lib.tf_run.restype = C.c_int64
    return _lib


def _arr(t) -> np.ndarray:
    a = t.detach().cpu().to(torch.float32).numpy() if isinstance(t, torch.Tensor) else t
    return np.ascontiguousarray(a, dtype=np.float32)


class NativeCPU:
    """One machine. load(program | state | tensor), then step(n).

    routed=True (default) evaluates only the MLP units that tfcpu.routing proves can be non-zero
    for the decoded instruction; routed=False evaluates every unit. Both are bit-identical.
    """

    ERRORS = {-1: "fetch tie", -2: "read-head tie", -3: "store-port tie"}
    OUT_CAP = 1 << 16

    def __init__(self, model: LoopedTransformer, routed: bool = True):
        if getattr(model, "arch", None) != "v2":
            raise ValueError("NativeCPU needs the v2 architecture")
        self.model = model.float() if next(model.parameters()).dtype != torch.float32 else model
        self.L = model.layout
        self._keep = []                       # numpy buffers referenced by the C struct
        b1, b2 = self.model.blocks
        a1, a2 = b1.attn, b2.attn
        self._static = dict(
            q1=self._csr(a1.q.weight), o1=self._csr(a1.o.weight),
            m1a=self._csr(b1.mlp.fc1.weight), m1b=self._csr(b1.mlp.fc2.weight),
            c1a=self._ptr(_arr(b1.mlp.fc1.bias)), c1b=self._ptr(_arr(b1.mlp.fc2.bias)),
            q2=self._csr(a2.q.weight), k2=self._csr(a2.k.weight), v2=self._csr(a2.v.weight), o2=self._csr(a2.o.weight),
            m2a=self._csr(b2.mlp.fc1.weight), m2b=self._csr(b2.mlp.fc2.weight),
            c2a=self._ptr(_arr(b2.mlp.fc1.bias)), c2b=self._ptr(_arr(b2.mlp.fc2.bias)),
            dk1=a1.d_k, dv1=a1.d_v, H=a2.n_heads, R=getattr(a2, "n_read_heads", 1), dk2=a2.d_k, dv2=a2.d_v,
            m1bT=self._csr(b1.mlp.fc2.weight.T), m2bT=self._csr(b2.mlp.fc2.weight.T),
            f_op0=self.L["f_op0"], f_rd0=self.L["f_rd0"], f_ra0=self.L["f_ra0"], f_rb0=self.L["f_rb0"],
            nops=NOPS, nreg=NREG,
        )
        self.routed = routed
        if routed:
            r = route(self.model)
            lists1, lists2, mem = r["b1"], r["b2"], r["mem"]
        else:
            all1 = np.arange(b1.mlp.fc1.out_features, dtype=np.int32)
            all2 = np.arange(b2.mlp.fc1.out_features, dtype=np.int32)
            lists1, lists2, mem = [all1] * NCOMBO, [all2] * NCOMBO, all2
        self.live = {"b1": np.array([len(x) for x in lists1]), "b2": np.array([len(x) for x in lists2]),
                     "mem": len(mem), "h1": b1.mlp.fc1.out_features, "h2": b2.mlp.fc1.out_features}
        for name, lists in (("r1", lists1), ("r2", lists2)):
            off = np.concatenate([[0], np.cumsum([len(x) for x in lists])]).astype(np.int32)
            idx = np.concatenate(lists).astype(np.int32)
            self._static[f"{name}_off"], self._static[f"{name}_idx"] = self._ptr(off, i32p), self._ptr(idx, i32p)
        self._static["rm_idx"], self._static["rm_n"] = self._ptr(np.ascontiguousarray(mem, dtype=np.int32), i32p), len(mem)
        self._lib = _load_lib()
        self.x_np: np.ndarray | None = None

    def _ptr(self, a: np.ndarray, kind=f32p):
        self._keep.append(a)
        return a.ctypes.data_as(kind)

    def _csr(self, W) -> _CSR:
        W = _arr(W)
        rows, cols = W.shape
        nz = W != 0
        indptr = np.concatenate([[0], np.cumsum(nz.sum(1))]).astype(np.int32)
        r, c = np.nonzero(nz)
        return _CSR(rows, cols, self._ptr(indptr, i32p), self._ptr(c.astype(np.int32), i32p),
                    self._ptr(W[r, c].astype(np.float32)))

    def load(self, what: Program | State | torch.Tensor) -> None:
        if isinstance(what, Program):
            what = initial_state(what)
        x = encode(what, self.L, torch.float32) if isinstance(what, State) else what.to(torch.float32).cpu()
        f = FastCPU(self.model)
        f.load(x)                                            # reuse FastCPU's cache construction
        self.x_np = np.ascontiguousarray(x.numpy().copy())
        self.x = torch.from_numpy(self.x_np)                 # shares memory: always current
        nmem = NMEM
        self._dyn = [_arr(f.K1[0]), _arr(f.V1[0]), _arr(f.K2read[0]), _arr(f.V2read[0]),
                     _arr(f.Qstore[0]), _arr(f.best_other[0])]
        K1, V1, K2, V2, Qs, best = self._dyn
        self._m = _Model(T=NTOK, d=self.L.d, cpu=CPU, mem0=MEM0, nmem=nmem, halted=self.L["halted"],
                         out=self.L["out"], out_flag=self.L["out_flag"],
                         K1=K1.ctypes.data_as(f32p), V1=V1.ctypes.data_as(f32p),
                         K2=K2.ctypes.data_as(f32p), V2=V2.ctypes.data_as(f32p),
                         Qs=Qs.ctypes.data_as(f32p), best=best.ctypes.data_as(f32p), **self._static)
        self._xp = self.x_np.ctypes.data_as(f32p)
        self._outbuf = np.zeros(self.OUT_CAP, dtype=np.float32)
        self._nout = C.c_int64(0)
        self.output: list[int] = []

    def step(self, n: int = 1) -> int:
        """Run up to n steps (stopping after a HALT); returns how many ran. OUT values go to .output."""
        self._nout.value = 0
        k = self._lib.tf_run(C.byref(self._m), self._xp, n,
                             self._outbuf.ctypes.data_as(f32p), self.OUT_CAP, C.byref(self._nout))
        if k < 0:
            raise RuntimeError(f"v2 invariant violated: {self.ERRORS.get(k, k)}")
        got = self._nout.value
        if got > self.OUT_CAP:
            raise RuntimeError(f"more than {self.OUT_CAP} OUT values in one step() call")
        self.output.extend(int(v) for v in self._outbuf[:got])
        return int(k)

    def state(self) -> State:
        return decode(self.x, self.L)


def run_native(model: LoopedTransformer, program: Program, max_steps: int = 10_000) -> tuple[State, int]:
    """Run one program to HALT (or max_steps); returns (final state, steps executed). See NativeCPU.output."""
    e = NativeCPU(model)
    e.load(program)
    steps = e.step(max_steps)
    return e.state(), steps
