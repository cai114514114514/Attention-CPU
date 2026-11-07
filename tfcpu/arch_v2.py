"""Architecture v2: the same machine in two blocks, exact in float32 (spec §7).

  B1  attn: CPU fetches PROG[pc]
      mlp : everything the step needs, speculatively, in one layer
            ea / sa   = clamp(R[ra] + imm)   (select and clamp fused: gated clamps)
            sv        = R[rb]
            add_*/sub_* raw R[ra] +- R[rb] and its overflow bits, for all 16 (ra, rb) pairs
            take      = JMP or (JZ and R[ra]==0) or (JNEG and R[ra]<0)
            jt, npc, sen
  B2  attn: head 1  CPU reads MEM[ea]                         (load port)
            head 2  MEM token i picks the CPU iff sa >= a_i   (store port: two comparators,
            head 3  MEM token i picks the CPU iff sa <= a_i    their AND is the address decoder)
      mlp : write registers / pc / halted / val; clear scratch

Two attention layers is the minimum: the memory read depends on the fetched
instruction. Wrap-around is split across the two MLPs (overflow bits in B1,
x 65536 in B2), so every hidden activation times its output weight stays below
2^24 and float32 is exact.
"""
from __future__ import annotations

from .gadgets import F, MLPBuilder, all_of, clamp, copy, gate, is_neg, is_zero, step_ge
from .isa import NMEM, NREG, Op
from .layout import Layout
from .nn import Block
from .wiring import ATT_BIG, FETCH_FIELDS, PERSISTENT, Head, address_match, attention

HALF = 1 << 15
WRAP = 1 << 16


def make_layout() -> Layout:
    scratch = (
        [f"f_{n}" for n in FETCH_FIELDS]                                    # B1 attn
        + ["ea", "sa", "sen", "sv",                                          # B1 mlp
           "add_raw", "add_hi", "add_lo", "sub_raw", "sub_hi", "sub_lo",
           "take", "jt", "npc",
           "mv", "rx_sv", "got_ge", "got_le"]                                # B2 attn
    )
    return Layout(PERSISTENT, scratch)


LAYOUT = make_layout()


def _op(o: Op):
    return F(f"f_op{int(o)}")


def _b1(L):
    attn = attention(L, [Head(address_match("pc", "is_prog"), [(n, f"f_{n}") for n in FETCH_FIELDS])])
    b = MLPBuilder(L)
    running_cpu = [F("is_cpu"), 1 - F("halted")]
    for i in range(NREG):
        R, ra, rb = F(f"R{i}"), F(f"f_ra{i}"), F(f"f_rb{i}")
        clamp(b, R + F("f_imm"), 0, NMEM - 1, "ea", conds=[ra])
        clamp(b, R + F("f_imm"), 0, NMEM - 1, "sa", conds=[ra, _op(Op.STORE), *running_cpu])
        gate(b, R, [rb], "sv")
        is_zero(b, R, "take", conds=[ra, _op(Op.JZ)])
        is_neg(b, R, "take", conds=[ra, _op(Op.JNEG)])
        for j in range(NREG):
            pair = [ra, F(f"f_rb{j}")]
            for name, x in (("add", R + F(f"R{j}")), ("sub", R - F(f"R{j}"))):
                gate(b, x, pair, f"{name}_raw")
                step_ge(b, x, HALF, f"{name}_hi", conds=pair)          # raw >= 32768
                step_ge(b, -x, HALF + 1, f"{name}_lo", conds=pair)     # raw < -32768
    b.relu(_op(Op.JMP), {"take": 1.0})
    all_of(b, [_op(Op.STORE), *running_cpu], "sen")
    clamp(b, F("f_imm"), 0, NMEM - 1, "jt")
    clamp(b, F("pc") + 1, 0, NMEM - 1, "npc")
    return Block(attn, b.build(), "B1 fetch / speculative execute")


STORE_OFF = 4096   # pushes the CPU's store-port score negative when it is not storing


def _store_heads() -> list[Head]:
    """Two comparator heads; MEM token i is the store target iff both pick the CPU.

    For a MEM query the CPU key scores ATT_BIG + (sa - a_i + 1/2) [head ge] or
    ATT_BIG + (a_i - sa + 1/2) [head le], minus STORE_OFF unless the CPU is storing.
    Every MEM key scores exactly ATT_BIG, so the CPU wins head ge iff sa >= a_i and
    head le iff sa <= a_i (scores differ by >= 1/2: never a tie with the CPU). When
    the CPU loses, the query averages over the 64 tied MEM keys, whose copied
    values (sv, is_cpu) are all exactly 0. PROG keys score 0 and never win.

    A score that peaks only at sa == a_i would need sa^2 in the key, which is not
    linear in the residual stream; two linear comparators avoid that.
    """
    common = [
        (F("one") * STORE_OFF, F("sen") - F("is_cpu")),           # CPU: -STORE_OFF unless storing
        (F("one") * ATT_BIG, F("is_mem") + F("is_cpu")),          # MEM and CPU keys only
    ]
    ge = Head([(F("one"), F("sa") + F("is_cpu") * 0.5), (F("a") * -1, F("is_cpu")), *common],
              [("is_cpu", "got_ge"), ("sv", "rx_sv")])
    le = Head([(F("one"), F("is_cpu") * 0.5 - F("sa")), (F("a"), F("is_cpu")), *common],
              [("is_cpu", "got_le")])
    return [ge, le]


def _b2(L):
    attn = attention(L, [Head(address_match("ea", "is_mem"), [("val", "mv")]), *_store_heads()])
    b = MLPBuilder(L)
    live = [F("is_cpu"), 1 - F("halted")]
    add = F("add_raw") - F("add_hi") * WRAP + F("add_lo") * WRAP
    sub = F("sub_raw") - F("sub_hi") * WRAP + F("sub_lo") * WRAP
    for i in range(NREG):
        R, rd = F(f"R{i}"), F(f"f_rd{i}")
        for o, value in ((Op.LOADI, F("f_imm")), (Op.LOAD, F("mv")), (Op.ADD, add), (Op.SUB, sub)):
            gate(b, value - R, [_op(o), rd, *live], f"R{i}")
    gate(b, F("npc") - F("pc"), [*live, 1 - _op(Op.HALT), 1 - F("take")], "pc")
    gate(b, F("jt") - F("pc"), [*live, F("take")], "pc")
    all_of(b, [*live, _op(Op.HALT)], "halted")
    gate(b, F("rx_sv") - F("val"), [F("got_ge"), F("got_le"), F("is_mem")], "val")
    for name in L.scratch:
        copy(b, F(name), name, coef=-1.0)                    # scratch := 0
    return Block(attn, b.build(), "B2 load / store / write back")


BLOCKS = (_b1, _b2)
