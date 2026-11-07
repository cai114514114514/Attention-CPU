import pytest

from tfcpu.isa import NMEM, NPROG, Instr, Op, Program, clamp_a, wrap
from tfcpu.reference import initial_state, step_ref


def make(instrs, regs=(0, 0, 0, 0), pc=0, halted=False, mem=None):
    prog = Program.from_list(instrs, mem)
    s = initial_state(prog)
    return s.replace(regs=tuple(regs), pc=pc, halted=halted)


def test_wrap_and_clamp():
    assert wrap(32767) == 32767
    assert wrap(32768) == -32768
    assert wrap(-32769) == 32767
    assert wrap(65534) == -2
    assert wrap(-65536) == 0
    assert clamp_a(-5) == 0 and clamp_a(70) == 63 and clamp_a(17) == 17


def test_program_padding_and_validation():
    p = Program.from_list([Instr(Op.LOADI, rd=1, imm=5)])
    assert len(p.instrs) == NPROG and len(p.mem) == NMEM
    assert p.instrs[1] == Instr(Op.HALT)
    with pytest.raises(ValueError):
        Instr(Op.ADD, rd=4)
    with pytest.raises(ValueError):
        Instr(Op.LOADI, imm=40000)
    with pytest.raises(ValueError):
        Program.from_list([Instr(Op.HALT)] * 65)


def test_loadi():
    s = step_ref(make([Instr(Op.LOADI, rd=2, imm=-7)]))
    assert s.regs == (0, 0, -7, 0) and s.pc == 1


def test_load_with_offset_and_clamp():
    mem = [0] * NMEM
    mem[5], mem[0], mem[63] = 11, 22, 33
    s = step_ref(make([Instr(Op.LOAD, rd=0, ra=1, imm=3)], regs=(0, 2, 0, 0), mem=mem))
    assert s.regs[0] == 11
    s = step_ref(make([Instr(Op.LOAD, rd=0, ra=1, imm=-10)], regs=(0, 2, 0, 0), mem=mem))
    assert s.regs[0] == 22
    s = step_ref(make([Instr(Op.LOAD, rd=0, ra=1, imm=100)], regs=(0, 2, 0, 0), mem=mem))
    assert s.regs[0] == 33


def test_store():
    s = step_ref(make([Instr(Op.STORE, rb=3, ra=1, imm=1)], regs=(0, 4, 0, -9)))
    assert s.mem[5] == -9 and s.pc == 1
    s = step_ref(make([Instr(Op.STORE, rb=3, ra=1, imm=1)], regs=(0, 30000, 0, 8)))
    assert s.mem[63] == 8


def test_add_sub_wrap():
    s = step_ref(make([Instr(Op.ADD, rd=0, ra=1, rb=2)], regs=(0, 32767, 1, 0)))
    assert s.regs[0] == -32768
    s = step_ref(make([Instr(Op.SUB, rd=3, ra=1, rb=2)], regs=(0, -32768, 1, 0)))
    assert s.regs[3] == 32767


def test_jumps():
    j = [Instr(Op.JZ, ra=1, imm=40)]
    assert step_ref(make(j, regs=(0, 0, 0, 0))).pc == 40
    assert step_ref(make(j, regs=(0, 5, 0, 0))).pc == 1
    n = [Instr(Op.JNEG, ra=2, imm=-3)]
    assert step_ref(make(n, regs=(0, 0, -1, 0))).pc == 0
    assert step_ref(make(n, regs=(0, 0, 0, 0))).pc == 1
    assert step_ref(make([Instr(Op.JMP, imm=99)])).pc == 63


def test_pc_saturates_at_63():
    instrs = [Instr(Op.LOADI, rd=0, imm=1)] * NPROG
    assert step_ref(make(instrs, pc=63)).pc == 63


def test_halt_is_sticky():
    s = step_ref(make([Instr(Op.HALT)], pc=0))
    assert s.halted and s.pc == 0
    s2 = step_ref(make([Instr(Op.LOADI, rd=0, imm=1)], halted=True, regs=(1, 2, 3, 4)))
    assert s2.halted and s2.regs == (1, 2, 3, 4) and s2.pc == 0


# ---- extended ISA (ops 9..16) -------------------------------------------------
import random as _random

from tfcpu.isa import BASE_OPS, SP_EMPTY, mul16
from tfcpu.reference import run_ref, outputs


def test_base_ops_are_the_original_nine():
    assert BASE_OPS == frozenset(Op(k) for k in range(9))


def test_addi_wraps():
    s = step_ref(make([Instr(Op.ADDI, rd=1, ra=2, imm=5)], regs=(0, 0, 32765, 0)))
    assert s.regs[1] == -32766 and s.pc == 1


def test_jr_clamps():
    assert step_ref(make([Instr(Op.JR, ra=0)], regs=(17, 0, 0, 0))).pc == 17
    assert step_ref(make([Instr(Op.JR, ra=0)], regs=(-5, 0, 0, 0))).pc == 0
    assert step_ref(make([Instr(Op.JR, ra=0)], regs=(999, 0, 0, 0))).pc == 63


def test_call_ret_push_pop():
    s = make([Instr(Op.CALL, imm=10)], pc=0)
    assert s.sp == SP_EMPTY == 64
    s = step_ref(s)
    assert (s.pc, s.sp, s.mem[63]) == (10, 63, 1)
    s = step_ref(s.replace(program=(Instr(Op.RET),) * 64))
    assert (s.pc, s.sp) == (1, 64)
    s = step_ref(make([Instr(Op.PUSH, rb=2)], regs=(0, 0, -7, 0)))
    assert (s.sp, s.mem[63]) == (63, -7)
    s = step_ref(s.replace(program=(Instr(Op.POP, rd=3),) * 64))
    assert (s.sp, s.regs[3]) == (64, -7)


def test_stack_pointer_saturates():
    s = step_ref(make([Instr(Op.PUSH, rb=0)], regs=(5, 0, 0, 0)).replace(sp=0))
    assert (s.sp, s.mem[0]) == (0, 5)
    s = step_ref(make([Instr(Op.POP, rd=0)], mem=[0] * 63 + [9]))
    assert (s.sp, s.regs[0]) == (64, 9)
    s = step_ref(make([Instr(Op.RET)], mem=[0] * 63 + [500]).replace(sp=63))
    assert s.pc == 63                                  # return address is clamped


def test_out_flag_is_one_step_only():
    s = step_ref(make([Instr(Op.OUT, ra=1), Instr(Op.LOADI, rd=0, imm=1)], regs=(0, 42, 0, 0)))
    assert (s.out, s.out_flag) == (42, True)
    s = step_ref(s)
    assert (s.out, s.out_flag) == (42, False)


def test_mul_takes_17_steps_and_wraps():
    rng = _random.Random(5)
    pairs = [(3, 4), (-3, 4), (-3, -4), (32767, 2), (-32768, -1), (255, 257), (0, -9)]
    pairs += [(rng.randint(-32768, 32767), rng.randint(-32768, 32767)) for _ in range(300)]
    for a, b in pairs:
        s = make([Instr(Op.MUL, rd=2, ra=0, rb=1)], regs=(a, b, 0, 0))
        for k in range(16):
            s = step_ref(s)
            assert s.pc == 0 and s.mstep == k + 1
        s = step_ref(s)
        assert (s.pc, s.mstep, s.regs[2]) == (1, 0, mul16(a, b)), (a, b)
    assert mul16(-32768, -1) == -32768 and mul16(300, 300) == 90000 - 65536


def test_halted_freezes_extended_state():
    s = make([Instr(Op.OUT, ra=0)], halted=True).replace(out=3, out_flag=True, mstep=4)
    assert step_ref(s) == s


def test_outputs_helper():
    prog = Program.from_list([Instr(Op.LOADI, rd=0, imm=7), Instr(Op.OUT, ra=0), Instr(Op.OUT, ra=0)])
    assert outputs(run_ref(prog)) == [7, 7]
