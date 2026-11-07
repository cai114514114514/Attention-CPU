"""Architecture v2: the same machine in two blocks, exact in float32 (spec §7).

  B1  attn: CPU fetches PROG[pc]
      mlp : everything the step needs, speculatively, in one layer
            ea / sa   = clamp(R[ra] + imm)   (select and clamp fused: gated clamps)
            sv        = R[rb]
            add_*/sub_* raw R[ra] +- R[rb] and its overflow bits, for all 16 (ra, rb) pairs
            take      = JMP or (JZ and R[ra]==0) or (JNEG and R[ra]<0)
            jt, npc, sen
            extended ISA (§8): ADDI raw/overflow per register, JR target, SP-1 / SP+1,
            MUL micro-step: [mstep==0], [mstep==16], [mb<0], wrap bits of 2*macc(+ma), 2*mb
  B2  attn: head 1  CPU reads MEM[ea]                         (load port)
            head 2  MEM token i picks the CPU iff sa >= a_i   (store port: two comparators,
            head 3  MEM token i picks the CPU iff sa <= a_i    their AND is the address decoder)
      mlp : write registers / pc / halted / sp / out / MUL temps / val; clear scratch

MUL is multi-cycle (like microcoded MUL on real CPUs): one setup step copies the
operands into ma/mb, then 16 steps of  macc = wrap(2*macc + (mb<0 ? ma : 0)),
mb = wrap(2*mb)  (MSB-first shift-and-add); pc holds until the last step, which
writes rd. Each step needs only a sign test, wrap bits and a 2-way select, so it
fits the same two blocks.

Two attention layers is the minimum: the memory read depends on the fetched
instruction. Wrap-around is split across the two MLPs (overflow bits in B1,
x 65536 in B2), so every hidden activation times its output weight stays below
2^24 and float32 is exact.
"""
from __future__ import annotations

from .gadgets import F, MLPBuilder, all_of, clamp, copy, gate, is_neg, is_zero, step_ge
from .isa import MUL_STEPS, NMEM, NREG, SP_EMPTY, Op
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
           "sel_a", "jr", "addi_raw", "addi_hi", "addi_lo", "spm", "spp",    # extended ISA
           "mul_setup", "mul_done", "mneg", "c0_hi", "c0_lo", "c1_hi", "c1_lo", "mb_hi", "mb_lo",
           "mv", "rx_sv", "got_ge", "got_le"]                                # B2 attn
    )
    return Layout(PERSISTENT, scratch)


LAYOUT = make_layout()


def _op(o: Op):
    return F(f"f_op{int(o)}")


def _wrap_bits(b: MLPBuilder, x, name: str) -> None:
    """Overflow bits of an integer x in [-98304, 98303]: name_hi = [x >= 2^15], name_lo = [x < -2^15]."""
    step_ge(b, x, HALF, f"{name}_hi")
    step_ge(b, -x, HALF + 1, f"{name}_lo")


def _b1(L):
    attn = attention(L, [Head(address_match("pc", "is_prog"), [(n, f"f_{n}") for n in FETCH_FIELDS])])
    b = MLPBuilder(L)
    running_cpu = [F("is_cpu"), 1 - F("halted")]
    stack_load = _op(Op.POP) + _op(Op.RET)           # one-hot sums of op flags are themselves 0/1
    stack_store = _op(Op.PUSH) + _op(Op.CALL)
    for i in range(NREG):
        R, ra, rb = F(f"R{i}"), F(f"f_ra{i}"), F(f"f_rb{i}")
        clamp(b, R + F("f_imm"), 0, NMEM - 1, "ea", conds=[ra, 1 - stack_load])
        clamp(b, R + F("f_imm"), 0, NMEM - 1, "sa", conds=[ra, _op(Op.STORE), *running_cpu])
        gate(b, R, [rb, 1 - _op(Op.CALL)], "sv")
        gate(b, R, [ra], "sel_a")
        clamp(b, R, 0, NMEM - 1, "jr", conds=[ra])
        is_zero(b, R, "take", conds=[ra, _op(Op.JZ)])
        is_neg(b, R, "take", conds=[ra, _op(Op.JNEG)])
        x = R + F("f_imm")                                            # ADDI, speculatively per register
        gate(b, x, [ra], "addi_raw")
        step_ge(b, x, HALF, "addi_hi", conds=[ra])
        step_ge(b, -x, HALF + 1, "addi_lo", conds=[ra])
        for j in range(NREG):
            pair = [ra, F(f"f_rb{j}")]
            for name, x in (("add", R + F(f"R{j}")), ("sub", R - F(f"R{j}"))):
                gate(b, x, pair, f"{name}_raw")
                step_ge(b, x, HALF, f"{name}_hi", conds=pair)          # raw >= 32768
                step_ge(b, -x, HALF + 1, f"{name}_lo", conds=pair)     # raw < -32768
    b.relu(_op(Op.JMP), {"take": 1.0})
    clamp(b, F("sp"), 0, NMEM - 1, "ea", conds=[stack_load])                       # POP / RET read M[SP]
    clamp(b, F("sp") - 1, 0, NMEM - 1, "sa", conds=[stack_store, *running_cpu])     # PUSH / CALL write M[SP-1]
    clamp(b, F("pc") + 1, 0, NMEM - 1, "sv", conds=[_op(Op.CALL)])                  # CALL pushes pc+1
    all_of(b, [_op(Op.STORE) + stack_store, *running_cpu], "sen")
    clamp(b, F("f_imm"), 0, NMEM - 1, "jt")
    clamp(b, F("pc") + 1, 0, NMEM - 1, "npc")
    clamp(b, F("sp") - 1, 0, SP_EMPTY, "spm")
    clamp(b, F("sp") + 1, 0, SP_EMPTY, "spp")
    # MUL micro-step
    is_zero(b, F("mstep"), "mul_setup", conds=[_op(Op.MUL)])
    is_zero(b, F("mstep") - MUL_STEPS, "mul_done", conds=[_op(Op.MUL)])
    is_neg(b, F("mb"), "mneg")
    _wrap_bits(b, F("macc") * 2, "c0")
    _wrap_bits(b, F("macc") * 2 + F("ma"), "c1")
    _wrap_bits(b, F("mb") * 2, "mb")
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


def _wrapped(raw, name: str):
    return raw - F(f"{name}_hi") * WRAP + F(f"{name}_lo") * WRAP


def _b2(L):
    attn = attention(L, [Head(address_match("ea", "is_mem"), [("val", "mv")]), *_store_heads()])
    b = MLPBuilder(L)
    live = [F("is_cpu"), 1 - F("halted")]
    add = _wrapped(F("add_raw"), "add")
    sub = _wrapped(F("sub_raw"), "sub")
    addi = _wrapped(F("addi_raw"), "addi")
    c0 = _wrapped(F("macc") * 2, "c0")                       # 2*macc
    c1 = _wrapped(F("macc") * 2 + F("ma"), "c1")             # 2*macc + ma
    mb2 = _wrapped(F("mb") * 2, "mb")
    mul_iter = _op(Op.MUL) - F("mul_setup")                  # MUL with mstep in 1..16
    mul_hold = _op(Op.MUL) - F("mul_done")                   # MUL that is not finishing: pc holds
    # registers
    sources = ((_op(Op.LOADI), F("f_imm")), (_op(Op.LOAD) + _op(Op.POP), F("mv")),
               (_op(Op.ADD), add), (_op(Op.SUB), sub), (_op(Op.ADDI), addi))
    for i in range(NREG):
        R, rd = F(f"R{i}"), F(f"f_rd{i}")
        for cond, value in sources:
            gate(b, value - R, [cond, rd, *live], f"R{i}")
        gate(b, c1 - R, [F("mul_done"), F("mneg"), rd, *live], f"R{i}")
        gate(b, c0 - R, [F("mul_done"), 1 - F("mneg"), rd, *live], f"R{i}")
    # pc: x_new = target*cond - x*cond, split into independent gated terms
    advance = 1 - _op(Op.HALT) - F("take") - _op(Op.JR) - _op(Op.CALL) - _op(Op.RET) - mul_hold
    gate(b, F("npc") - F("pc"), [*live, advance], "pc")
    gate(b, F("jt") - F("pc"), [*live, F("take") + _op(Op.CALL)], "pc")
    gate(b, F("jr") - F("pc"), [*live, _op(Op.JR)], "pc")
    clamp(b, F("mv"), 0, NMEM - 1, "pc", conds=[*live, _op(Op.RET)])     # RET: pc = clamp(M[SP])
    gate(b, -F("pc"), [*live, _op(Op.RET)], "pc")
    all_of(b, [*live, _op(Op.HALT)], "halted")
    # stack pointer
    gate(b, F("spm") - F("sp"), [*live, _op(Op.PUSH) + _op(Op.CALL)], "sp")
    gate(b, F("spp") - F("sp"), [*live, _op(Op.POP) + _op(Op.RET)], "sp")
    # output port
    gate(b, F("sel_a") - F("out"), [*live, _op(Op.OUT)], "out")
    gate(b, 1 - F("out_flag"), [*live, _op(Op.OUT)], "out_flag")
    gate(b, -F("out_flag"), [*live, 1 - _op(Op.OUT)], "out_flag")
    # MUL temporaries
    gate(b, F("sel_a") - F("ma"), [*live, F("mul_setup")], "ma")
    gate(b, F("sv") - F("mb"), [*live, F("mul_setup")], "mb")
    gate(b, -F("macc"), [*live, F("mul_setup")], "macc")
    gate(b, 1 - F("mstep"), [*live, F("mul_setup")], "mstep")
    gate(b, c1 - F("macc"), [*live, mul_iter, F("mneg")], "macc")
    gate(b, c0 - F("macc"), [*live, mul_iter, 1 - F("mneg")], "macc")
    gate(b, mb2 - F("mb"), [*live, mul_iter], "mb")
    gate(b, F("one"), [*live, mul_iter - F("mul_done")], "mstep")         # 1..15 -> +1
    gate(b, -F("mstep"), [*live, F("mul_done")], "mstep")                 # 16 -> 0
    # memory
    gate(b, F("rx_sv") - F("val"), [F("got_ge"), F("got_le"), F("is_mem")], "val")
    for name in L.scratch:
        copy(b, F(name), name, coef=-1.0)                    # scratch := 0
    return Block(attn, b.build(), "B2 load / store / write back")


BLOCKS = (_b1, _b2)
