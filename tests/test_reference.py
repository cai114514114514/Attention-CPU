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
