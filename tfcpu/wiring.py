"""Shared wiring helpers: the persistent fields and hand-built attention heads."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .gadgets import BIG, Aff, F
from .isa import NOPS, NREG
from .layout import Layout
from .nn import HardmaxAttention

ATT_BIG = BIG   # token-type bias in attention scores; any value >= 1 separates token types

# Persistent fields are identical in every architecture, so encode/decode work for all.
PERSISTENT = (
    ["one", "is_cpu", "is_prog", "is_mem", "a", "a2"]
    + [f"op{k}" for k in range(NOPS)]
    + [f"{f}{i}" for f in ("rd", "ra", "rb") for i in range(NREG)]
    + ["imm", "val"]
    + [f"R{i}" for i in range(NREG)]
    + ["pc", "halted"]
    + ["sp", "out", "out_flag", "mstep", "ma", "mb", "macc"]      # extended ISA (§8)
)
FETCH_FIELDS = [f"op{k}" for k in range(NOPS)] + [f"{f}{i}" for f in ("rd", "ra", "rb") for i in range(NREG)] + ["imm"]


@dataclass
class Head:
    """One attention head: score = sum_i q_i(x_query) * k_i(x_key); copies (src -> dst) fields."""

    dims: list[tuple[Aff, Aff]] = field(default_factory=list)
    copies: list[tuple[str, str]] = field(default_factory=list)


def address_match(query: str | None, key_type: str | None) -> list[tuple[Aff, Aff]]:
    """Score t^2 - (t - a)^2 + ATT_BIG*[key_type]: uniquely maximal at the typed token with a == t."""
    dims = []
    if query is not None:
        dims += [(F(query) * 2, F("a")), (F("one") * -1, F("a2"))]
    if key_type is not None:
        dims += [(F("one") * ATT_BIG, F(key_type))]
    return dims


def attention(L: Layout, heads: list[Head]) -> HardmaxAttention:
    """Build a hardmax attention layer from head specs. No heads = an all-zero layer (adds exactly 0)."""
    heads = heads or [Head()]
    d_k = max([1] + [len(h.dims) for h in heads])
    d_v = max([1] + [len(h.copies) for h in heads])
    att = HardmaxAttention(L.d, d_k, d_v, len(heads)).double()
    Wq = torch.zeros(len(heads) * d_k, L.d, dtype=torch.float64)
    Wk = torch.zeros_like(Wq)
    Wv = torch.zeros(len(heads) * d_v, L.d, dtype=torch.float64)
    Wo = torch.zeros(L.d, len(heads) * d_v, dtype=torch.float64)
    for h, head in enumerate(heads):
        for i, (q, k) in enumerate(head.dims):
            for W, aff in ((Wq, q), (Wk, k)):
                if aff.c:
                    raise ValueError("attention projections have no bias; use F('one') for constants")
                for name, v in aff.terms.items():
                    W[h * d_k + i, L[name]] += v
        for j, (src, dst) in enumerate(head.copies):
            Wv[h * d_v + j, L[src]] = 1.0
            Wo[L[dst], h * d_v + j] = 1.0
    with torch.no_grad():
        att.q.weight.copy_(Wq); att.k.weight.copy_(Wk)
        att.v.weight.copy_(Wv); att.o.weight.copy_(Wo)
    return att
