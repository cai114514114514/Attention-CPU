"""Hand-wires the weights of the six-block looped Transformer (spec §4).

One forward pass executes one instruction:

  B1  attn: CPU fetches PROG[pc]          mlp: A = R[ra], B = R[rb]
  B2  attn: -                             mlp: ea, sum, dif, z, n, npc, jt
  B3  attn: CPU reads MEM[ea]             mlp: res, take, sen
  B4  attn: -                             mlp: write registers / pc / halted; sa
  B5  attn: every token reads the CPU     mlp: hit = [rx_sa == a] and is_mem
  B6  attn: -                             mlp: val update; clear all scratch

Every write to persistent state is gated by the owning token type, every
scratch field is written by exactly one sub-layer, and B6 zeroes all scratch,
so model(encode(s)) == encode(step_ref(s)) exactly.
"""
from __future__ import annotations

import torch

from .gadgets import BIG, F, MLPBuilder, all_of, clamp, copy, gate, is_neg, is_zero, wrap16
from .isa import NMEM, NOPS, NREG, Op
from .layout import Layout
from .nn import Block, HardmaxAttention, LoopedTransformer

ATT_BIG = BIG          # type bias in attention scores; any value >= 1 separates token types
NO_STORE = -1000       # store address meaning "no memory word matches"


def make_layout() -> Layout:
    persistent = (
        ["one", "is_cpu", "is_prog", "is_mem", "a", "a2"]
        + [f"op{k}" for k in range(NOPS)]
        + [f"{f}{i}" for f in ("rd", "ra", "rb") for i in range(NREG)]
        + ["imm", "val"]
        + [f"R{i}" for i in range(NREG)]
        + ["pc", "halted"]
    )
    scratch = (
        [f"f_op{k}" for k in range(NOPS)]
        + [f"f_{f}{i}" for f in ("rd", "ra", "rb") for i in range(NREG)]
        + ["f_imm", "A", "B",                                     # B1
           "ea", "sum", "dif", "z", "n", "npc", "jt",             # B2
           "mv", "res", "take", "sen",                            # B3
           "sa",                                                  # B4
           "rx_sa", "rx_sv", "hit"]                               # B5
    )
    return Layout(persistent, scratch)


LAYOUT = make_layout()


# ---- attention --------------------------------------------------------------

def _attn(L: Layout, query_addr: str | None, key_type: str | None, copies: list[tuple[str, str]]):
    """Hardmax attention that selects the `key_type` token whose address a == query_addr.

    score = 2*t*a - a^2 + ATT_BIG*[key_type] = t^2 - (t-a)^2 + ATT_BIG*[key_type],
    uniquely maximal at the typed token with a == t (t an integer in 0..63).
    With query_addr=None the score is just the type bias. copies: (src, dst) fields.
    """
    d_k, d_v = 3, max(len(copies), 1)
    att = HardmaxAttention(L.d, d_k, d_v).double()
    Wq = torch.zeros(d_k, L.d, dtype=torch.float64)
    Wk = torch.zeros(d_k, L.d, dtype=torch.float64)
    Wv = torch.zeros(d_v, L.d, dtype=torch.float64)
    Wo = torch.zeros(L.d, d_v, dtype=torch.float64)
    if query_addr is not None:
        Wq[0, L[query_addr]] = 2.0
        Wk[0, L["a"]] = 1.0
        Wq[1, L["one"]] = -1.0
        Wk[1, L["a2"]] = 1.0
    if key_type is not None:
        Wq[2, L["one"]] = ATT_BIG
        Wk[2, L[key_type]] = 1.0
    for j, (src, dst) in enumerate(copies):
        Wv[j, L[src]] = 1.0
        Wo[L[dst], j] = 1.0
    with torch.no_grad():
        att.q.weight.copy_(Wq); att.k.weight.copy_(Wk)
        att.v.weight.copy_(Wv); att.o.weight.copy_(Wo)
    return att


def _no_attn(L: Layout):
    """All-zero attention: contributes exactly 0 to the residual stream."""
    return _attn(L, None, None, [])


# ---- the six blocks ---------------------------------------------------------

def _op(o: Op):
    return F(f"f_op{int(o)}")


def _b1(L):
    fetch = [(f"op{k}", f"f_op{k}") for k in range(NOPS)]
    fetch += [(f"{f}{i}", f"f_{f}{i}") for f in ("rd", "ra", "rb") for i in range(NREG)]
    fetch += [("imm", "f_imm")]
    attn = _attn(L, "pc", "is_prog", fetch)
    b = MLPBuilder(L)
    for i in range(NREG):
        gate(b, F(f"R{i}"), [F(f"f_ra{i}")], "A")
        gate(b, F(f"R{i}"), [F(f"f_rb{i}")], "B")
    return Block(attn, b.build(), "B1 fetch / read registers")


def _b2(L):
    b = MLPBuilder(L)
    clamp(b, F("A") + F("f_imm"), 0, NMEM - 1, "ea")
    wrap16(b, F("A") + F("B"), "sum")
    wrap16(b, F("A") - F("B"), "dif")
    is_zero(b, F("A"), "z")
    is_neg(b, F("A"), "n")
    clamp(b, F("pc") + 1, 0, NMEM - 1, "npc")
    clamp(b, F("f_imm"), 0, NMEM - 1, "jt")
    return Block(_no_attn(L), b.build(), "B2 ALU")


def _b3(L):
    attn = _attn(L, "ea", "is_mem", [("val", "mv")])
    b = MLPBuilder(L)
    gate(b, F("f_imm"), [_op(Op.LOADI)], "res")
    gate(b, F("mv"), [_op(Op.LOAD)], "res")
    gate(b, F("sum"), [_op(Op.ADD)], "res")
    gate(b, F("dif"), [_op(Op.SUB)], "res")
    b.relu(_op(Op.JMP), {"take": 1.0})                       # op flags are 0/1
    all_of(b, [_op(Op.JZ), F("z")], "take")
    all_of(b, [_op(Op.JNEG), F("n")], "take")
    all_of(b, [_op(Op.STORE), F("is_cpu"), 1 - F("halted")], "sen")
    return Block(attn, b.build(), "B3 memory read / select result")


def _b4(L):
    b = MLPBuilder(L)
    live = [F("is_cpu"), 1 - F("halted")]                    # a running CPU token
    we = _op(Op.LOADI) + _op(Op.LOAD) + _op(Op.ADD) + _op(Op.SUB)
    for i in range(NREG):
        gate(b, F("res") - F(f"R{i}"), [we, F(f"f_rd{i}"), *live], f"R{i}")
    gate(b, F("npc") - F("pc"), [*live, 1 - _op(Op.HALT), 1 - F("take")], "pc")
    gate(b, F("jt") - F("pc"), [*live, F("take")], "pc")
    all_of(b, [*live, _op(Op.HALT)], "halted")
    gate(b, F("ea") - NO_STORE, [F("sen")], "sa")
    b.bias("sa", NO_STORE)
    return Block(_no_attn(L), b.build(), "B4 write back")


def _b5(L):
    attn = _attn(L, None, "is_cpu", [("sa", "rx_sa"), ("B", "rx_sv")])
    b = MLPBuilder(L)
    is_zero(b, F("rx_sa") - F("a"), "hit", conds=[F("is_mem")])
    return Block(attn, b.build(), "B5 broadcast store")


def _b6(L):
    b = MLPBuilder(L)
    gate(b, F("rx_sv") - F("val"), [F("hit")], "val")
    for name in L.scratch:
        copy(b, F(name), name, coef=-1.0)                    # scratch := 0
    return Block(_no_attn(L), b.build(), "B6 memory write / clear scratch")


def build_cpu(layout: Layout = LAYOUT) -> LoopedTransformer:
    """Build the CPU. Returns an ordinary float64 nn.Module; one forward = one step."""
    blocks = [f(layout) for f in (_b1, _b2, _b3, _b4, _b5, _b6)]
    model = LoopedTransformer(blocks).double().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.layout = layout
    return model
