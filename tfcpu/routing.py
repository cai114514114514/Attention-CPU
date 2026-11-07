"""Static routing for the v2 engine: prove, per decoded instruction, which MLP units are dead.

For every combination of (opcode, rd, ra, rb) the fetched one-hot fields are fixed, and every
other field on the CPU row has a known range (registers are 16-bit words, pc is 0..63, the
machine is running, ...). Interval arithmetic then gives an upper bound on each hidden unit's
pre-activation; a unit whose bound is <= 0 outputs exactly 0 for that instruction, so the
engine may skip it -- the result is bit-identical (relu(negative) == 0). This is a provably
correct mixture-of-experts router whose "experts" are individual neurons.

Ranges are propagated: B1's live units bound the scratch fields B2 reads, and B2's read heads
bound mv / amax. A MEM row being written gets its own (instruction-independent) live list.
"""
from __future__ import annotations

import numpy as np

from .isa import MUL_STEPS, NMEM, NOPS, NREG, NPROG, SP_EMPTY, WMAX, WMIN
from .layout import Layout

WORD = (WMIN, WMAX)


def _np(t) -> np.ndarray:
    return t.detach().cpu().double().numpy()


def _ub(W: np.ndarray, b: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Upper bound of W x + b over lo <= x <= hi (rows of lo/hi are separate boxes)."""
    return hi @ np.maximum(W, 0).T + lo @ np.minimum(W, 0).T + b


def _relu_out(W2: np.ndarray, b2: np.ndarray, ub: np.ndarray, lo: np.ndarray, hi: np.ndarray):
    """Ranges after x + W2 relu(.) + b2, with each relu in [0, max(ub, 0)]."""
    r = np.maximum(ub, 0)
    return lo + b2 + r @ np.minimum(W2, 0).T, hi + b2 + r @ np.maximum(W2, 0).T


def _box(L: Layout, n: int, **ranges) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.zeros((n, L.d)), np.zeros((n, L.d))
    for name, (a, b) in ranges.items():
        lo[:, L[name]], hi[:, L[name]] = a, b
    return lo, hi


def combo_index(op: int, rd: int, ra: int, rb: int) -> int:
    return ((op * NREG + rd) * NREG + ra) * NREG + rb


NCOMBO = NOPS * NREG ** 3


def route(model) -> dict:
    """Live unit lists: {'b1': [NCOMBO lists], 'b2': [NCOMBO lists], 'mem': list}."""
    L = model.layout
    b1, b2 = model.blocks
    W1a, c1a, W1b, c1b = _np(b1.mlp.fc1.weight), _np(b1.mlp.fc1.bias), _np(b1.mlp.fc2.weight), _np(b1.mlp.fc2.bias)
    W2a, c2a, W2b, c2b = _np(b2.mlp.fc1.weight), _np(b2.mlp.fc1.bias), _np(b2.mlp.fc2.weight), _np(b2.mlp.fc2.bias)
    a2 = b2.attn
    R, H, dv = getattr(a2, "n_read_heads", 1), a2.n_heads, a2.d_v
    Wv, Wo = _np(a2.v.weight), _np(a2.o.weight)

    # CPU row entering B1's MLP: a running machine, fetched fields pinned per combination
    cpu = dict(one=(1, 1), is_cpu=(1, 1), pc=(0, NPROG - 1), halted=(0, 0), sp=(0, SP_EMPTY),
               out=WORD, out_flag=(0, 1), mstep=(0, MUL_STEPS), ma=WORD, mb=WORD, macc=WORD, f_imm=WORD,
               **{f"R{i}": WORD for i in range(NREG)})
    lo, hi = _box(L, NCOMBO, **cpu)
    for op in range(NOPS):
        for rd in range(NREG):
            for ra in range(NREG):
                for rb in range(NREG):
                    k = combo_index(op, rd, ra, rb)
                    for f in (f"f_op{op}", f"f_rd{rd}", f"f_ra{ra}", f"f_rb{rb}"):
                        lo[k, L[f]] = hi[k, L[f]] = 1
    ub1 = _ub(W1a, c1a, lo, hi)
    lo, hi = _relu_out(W1b, c1b, ub1, lo, hi)

    # B2 read heads on the CPU row: each copies (through v, o) fields of one MEM row
    mlo, mhi = _box(L, 1, one=(1, 1), is_mem=(1, 1), a=(0, NMEM - 1), a2=(0, (NMEM - 1) ** 2), val=WORD)
    vlo = mlo @ np.maximum(Wv, 0).T + mhi @ np.minimum(Wv, 0).T             # lower bound of v(row)
    vhi = mhi @ np.maximum(Wv, 0).T + mlo @ np.minimum(Wv, 0).T
    keep = np.zeros(H * dv)
    keep[: R * dv] = 1                                                       # store heads are 0 on the CPU row
    vlo, vhi = vlo * keep, vhi * keep
    lo = lo + vlo @ np.maximum(Wo, 0).T + vhi @ np.minimum(Wo, 0).T
    hi = hi + vhi @ np.maximum(Wo, 0).T + vlo @ np.minimum(Wo, 0).T
    ub2 = _ub(W2a, c2a, lo, hi)

    # a MEM row being written: its store heads all picked the CPU (got = 1, rx_sv = R[rb] or pc+1)
    rlo, rhi = _box(L, 1, one=(1, 1), is_mem=(1, 1), a=(0, NMEM - 1), a2=(0, (NMEM - 1) ** 2), val=WORD,
                    rx_sv=WORD, got_ge=(1, 1), got_le=(1, 1))
    ubm = _ub(W2a, c2a, rlo, rhi)[0]

    return {
        "b1": [np.flatnonzero(u > 0).astype(np.int32) for u in ub1],
        "b2": [np.flatnonzero(u > 0).astype(np.int32) for u in ub2],
        "mem": np.flatnonzero(ubm > 0).astype(np.int32),
        "h1": W1a.shape[0], "h2": W2a.shape[0],
    }
