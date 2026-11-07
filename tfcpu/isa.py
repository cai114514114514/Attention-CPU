"""Instruction set and machine constants (spec §2, extended ISA §8)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

WORD = 16
WMIN, WMAX = -(1 << (WORD - 1)), (1 << (WORD - 1)) - 1
NREG = 4
NPROG = 64
NMEM = 64
SP_EMPTY = NMEM          # stack pointer value of an empty stack (grows down from the top of memory)
MUL_STEPS = 16           # MUL: one setup step + 16 shift-and-add steps


class Op(IntEnum):
    HALT = 0
    LOADI = 1
    LOAD = 2
    STORE = 3
    ADD = 4
    SUB = 5
    JZ = 6
    JNEG = 7
    JMP = 8
    # extended ISA
    ADDI = 9
    JR = 10
    CALL = 11
    RET = 12
    PUSH = 13
    POP = 14
    MUL = 15
    OUT = 16
    # vector instruction: one attention head scans all of memory
    ARGMAX = 17


NOPS = len(Op)
BASE_OPS = frozenset(Op(k) for k in range(9))      # the original nine (all that v1 implements)


def wrap(x: int) -> int:
    """Two's-complement 16-bit wrap-around."""
    return ((x - WMIN) % (1 << WORD)) + WMIN


def clamp_a(x: int) -> int:
    """Clamp an address / jump target to 0..63."""
    return min(max(x, 0), NMEM - 1)


def clamp_sp(x: int) -> int:
    """Clamp a stack pointer to 0..64."""
    return min(max(x, 0), SP_EMPTY)


def mul16(a: int, b: int) -> int:
    """Low 16 bits of a*b, as a signed word."""
    return wrap(a * b)


@dataclass(frozen=True)
class Instr:
    """One instruction. Unused fields are 0. STORE/PUSH keep rs in `rb`; JZ/JNEG/JR/OUT use `ra`."""

    op: Op
    rd: int = 0
    ra: int = 0
    rb: int = 0
    imm: int = 0

    def __post_init__(self):
        object.__setattr__(self, "op", Op(self.op))
        for name in ("rd", "ra", "rb"):
            v = getattr(self, name)
            if not 0 <= v < NREG:
                raise ValueError(f"{name}={v}: register must be 0..{NREG - 1}")
        if not WMIN <= self.imm <= WMAX:
            raise ValueError(f"imm={self.imm}: out of 16-bit range")

    def __str__(self) -> str:
        o, r = self.op, lambda i: f"r{i}"
        if o is Op.HALT:
            return "HALT"
        if o is Op.LOADI:
            return f"LOADI {r(self.rd)}, {self.imm}"
        if o is Op.LOAD:
            return f"LOAD  {r(self.rd)}, {r(self.ra)}, {self.imm}"
        if o is Op.STORE:
            return f"STORE {r(self.rb)}, {r(self.ra)}, {self.imm}"
        if o in (Op.ADD, Op.SUB):
            return f"{o.name:<5} {r(self.rd)}, {r(self.ra)}, {r(self.rb)}"
        if o in (Op.JZ, Op.JNEG):
            return f"{o.name:<5} {r(self.ra)}, {self.imm}"
        if o is Op.JMP:
            return f"JMP   {self.imm}"
        if o is Op.ADDI:
            return f"ADDI  {r(self.rd)}, {r(self.ra)}, {self.imm}"
        if o in (Op.JR, Op.OUT):
            return f"{o.name:<5} {r(self.ra)}"
        if o is Op.CALL:
            return f"CALL  {self.imm}"
        if o is Op.RET:
            return "RET"
        if o is Op.PUSH:
            return f"PUSH  {r(self.rb)}"
        if o is Op.POP:
            return f"POP   {r(self.rd)}"
        if o is Op.ARGMAX:
            return f"ARGMAX {r(self.rd)}"
        return f"MUL   {r(self.rd)}, {r(self.ra)}, {r(self.rb)}"


@dataclass(frozen=True)
class Program:
    """64 instructions (HALT-padded) plus the initial 64-word data memory."""

    instrs: tuple[Instr, ...]
    mem: tuple[int, ...]

    @classmethod
    def from_list(cls, instrs, mem=None) -> "Program":
        instrs = list(instrs)
        if len(instrs) > NPROG:
            raise ValueError(f"program has {len(instrs)} instructions; max is {NPROG}")
        instrs += [Instr(Op.HALT)] * (NPROG - len(instrs))
        mem = list(mem) if mem is not None else [0] * NMEM
        if len(mem) != NMEM:
            raise ValueError(f"memory must have {NMEM} words")
        for v in mem:
            if not WMIN <= v <= WMAX:
                raise ValueError(f"memory value {v} out of 16-bit range")
        return cls(tuple(instrs), tuple(int(v) for v in mem))
