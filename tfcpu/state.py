"""encode(State) -> tensor and decode(tensor) -> State (spec §3)."""
from __future__ import annotations

import numpy as np
import torch

from .build import LAYOUT
from .isa import MUL_STEPS, NMEM, NOPS, NPROG, NREG, SP_EMPTY, Instr, Op
from .layout import Layout
from .reference import State

CPU = 0
PROG0 = 1
MEM0 = 1 + NPROG
NTOK = 1 + NPROG + NMEM


class DecodeError(ValueError):
    pass


def encode(s: State, L: Layout = LAYOUT, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Canonical encoding: persistent fields set, scratch zero. Same persistent layout for every arch."""
    x = np.zeros((NTOK, L.d), dtype=np.float64)
    x[:, L["one"]] = 1.0
    x[CPU, L["is_cpu"]] = 1.0
    x[CPU, [L[f"R{i}"] for i in range(NREG)]] = s.regs
    x[CPU, L["pc"]] = s.pc
    x[CPU, L["halted"]] = float(s.halted)
    for f in ("sp", "out", "mstep", "ma", "mb", "macc"):
        x[CPU, L[f]] = getattr(s, f)
    x[CPU, L["out_flag"]] = float(s.out_flag)
    addr = np.arange(NPROG)
    rows = PROG0 + addr
    x[rows, L["is_prog"]] = 1.0
    x[rows, L["a"]] = addr
    x[rows, L["a2"]] = addr * addr
    prog = s.program
    x[rows, [L[f"op{int(i.op)}"] for i in prog]] = 1.0
    x[rows, [L[f"rd{i.rd}"] for i in prog]] = 1.0
    x[rows, [L[f"ra{i.ra}"] for i in prog]] = 1.0
    x[rows, [L[f"rb{i.rb}"] for i in prog]] = 1.0
    x[rows, L["imm"]] = [i.imm for i in prog]
    addr = np.arange(NMEM)
    rows = MEM0 + addr
    x[rows, L["is_mem"]] = 1.0
    x[rows, L["a"]] = addr
    x[rows, L["a2"]] = addr * addr
    x[rows, L["val"]] = s.mem
    return torch.from_numpy(x).to(dtype)


def _int(v: float, what: str) -> int:
    if v != int(v):
        raise DecodeError(f"{what} = {v} is not an integer")
    return int(v)


def _onehot(row: torch.Tensor, names: list[str], L: Layout, what: str) -> int:
    vals = [float(row[L[n]]) for n in names]
    if sorted(vals) != [0.0] * (len(vals) - 1) + [1.0]:
        raise DecodeError(f"{what} is not one-hot: {vals}")
    return vals.index(1.0)


def decode(x: torch.Tensor, L: Layout = LAYOUT) -> State:
    """Read the machine state back out; raises DecodeError if any invariant is broken."""
    x = x.detach().cpu()
    if tuple(x.shape) != (NTOK, L.d):
        raise DecodeError(f"expected shape {(NTOK, L.d)}, got {tuple(x.shape)}")
    if (x[:, [L[n] for n in L.scratch]] != 0).any():
        raise DecodeError("scratch fields are not all zero")
    cpu = x[CPU]
    regs = tuple(_int(float(cpu[L[f"R{i}"]]), f"R{i}") for i in range(NREG))
    pc = _int(float(cpu[L["pc"]]), "pc")
    halted = _int(float(cpu[L["halted"]]), "halted")
    if not 0 <= pc < NPROG or halted not in (0, 1):
        raise DecodeError(f"bad pc={pc} / halted={halted}")
    ext = {f: _int(float(cpu[L[f]]), f) for f in ("sp", "out", "out_flag", "mstep", "ma", "mb", "macc")}
    if not 0 <= ext["sp"] <= SP_EMPTY or ext["out_flag"] not in (0, 1) or not 0 <= ext["mstep"] <= MUL_STEPS:
        raise DecodeError(f"bad extended state {ext}")
    ext["out_flag"] = bool(ext["out_flag"])
    prog = []
    for j in range(NPROG):
        row = x[PROG0 + j]
        op = _onehot(row, [f"op{k}" for k in range(NOPS)], L, f"PROG[{j}].op")
        rd, ra, rb = (_onehot(row, [f"{f}{i}" for i in range(NREG)], L, f"PROG[{j}].{f}")
                      for f in ("rd", "ra", "rb"))
        prog.append(Instr(Op(op), rd, ra, rb, _int(float(row[L["imm"]]), f"PROG[{j}].imm")))
    mem = tuple(_int(float(x[MEM0 + j, L["val"]]), f"M[{j}]") for j in range(NMEM))
    out = State(tuple(prog), regs, pc, bool(halted), mem, **ext)
    if not torch.equal(encode(out, L, x.dtype), x):
        raise DecodeError("tensor is not a canonical encoding (stray values in foreign fields)")
    return out
