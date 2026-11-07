"""Architecture v1: the original six-block CPU (spec §4). Exact in float64.

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

from .gadgets import F, MLPBuilder, all_of, clamp, copy, gate, is_neg, is_zero, wrap16
from .isa import NMEM, NOPS, NREG, Op
from .layout import Layout
from .nn import Block
from .wiring import FETCH_FIELDS, PERSISTENT, Head, address_match, attention

NO_STORE = -1000       # store address meaning "no memory word matches"


def make_layout() -> Layout:
    scratch = (
        [f"f_{n}" for n in FETCH_FIELDS]
        + ["A", "B",                                              # B1
           "ea", "sum", "dif", "z", "n", "npc", "jt",             # B2
           "mv", "res", "take", "sen",                            # B3
           "sa",                                                  # B4
           "rx_sa", "rx_sv", "hit"]                               # B5
    )
    return Layout(PERSISTENT, scratch)


LAYOUT = make_layout()


def _op(o: Op):
    return F(f"f_op{int(o)}")


def _b1(L):
    attn = attention(L, [Head(address_match("pc", "is_prog"), [(n, f"f_{n}") for n in FETCH_FIELDS])])
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
    return Block(attention(L, []), b.build(), "B2 ALU")


def _b3(L):
    attn = attention(L, [Head(address_match("ea", "is_mem"), [("val", "mv")])])
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
    return Block(attention(L, []), b.build(), "B4 write back")


def _b5(L):
    attn = attention(L, [Head(address_match(None, "is_cpu"), [("sa", "rx_sa"), ("B", "rx_sv")])])
    b = MLPBuilder(L)
    is_zero(b, F("rx_sa") - F("a"), "hit", conds=[F("is_mem")])
    return Block(attn, b.build(), "B5 broadcast store")


def _b6(L):
    b = MLPBuilder(L)
    gate(b, F("rx_sv") - F("val"), [F("hit")], "val")
    for name in L.scratch:
        copy(b, F(name), name, coef=-1.0)                    # scratch := 0
    return Block(attention(L, []), b.build(), "B6 memory write / clear scratch")


BLOCKS = (_b1, _b2, _b3, _b4, _b5, _b6)
