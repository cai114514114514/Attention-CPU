import pytest

from tfcpu.asm import AsmError, assemble
from tfcpu.isa import Instr, Op


def test_basic_program_labels_comments_data():
    p = assemble("""
        ; demo
        start:  LOADI r1, 0x10   ; hex
                load  r0, r1, -2
        loop:   STORE r0, r2, 3
                ADD r3, r0, r1
                SUB r3, r3, r1
                JZ r3, end
                JNEG r0, loop
                JMP start
        end:    HALT
        .data 4 1 -2 3
        .data 63 7
    """)
    assert p.instrs[:9] == (
        Instr(Op.LOADI, rd=1, imm=16),
        Instr(Op.LOAD, rd=0, ra=1, imm=-2),
        Instr(Op.STORE, rb=0, ra=2, imm=3),
        Instr(Op.ADD, rd=3, ra=0, rb=1),
        Instr(Op.SUB, rd=3, ra=3, rb=1),
        Instr(Op.JZ, ra=3, imm=8),
        Instr(Op.JNEG, ra=0, imm=2),
        Instr(Op.JMP, imm=0),
        Instr(Op.HALT),
    )
    assert p.mem[4:7] == (1, -2, 3) and p.mem[63] == 7 and p.mem[0] == 0


def test_label_on_own_line():
    p = assemble("a:\n  JMP a\n")
    assert p.instrs[0] == Instr(Op.JMP, imm=0)


@pytest.mark.parametrize("src, msg", [
    ("FOO r1", "unknown"),
    ("JMP nowhere", "label"),
    ("LOADI r1, 99999", "range"),
    ("ADD r1, r2", "operand"),
    ("ADD r1, r2, r7", "register"),
    ("x:\nx:\nHALT", "duplicate"),
    ("HALT\n" * 65, "64"),
    (".data 62 1 2 3", "data"),
])
def test_errors(src, msg):
    with pytest.raises(AsmError, match=msg) as e:
        assemble(src)
    assert "line" in str(e.value)
