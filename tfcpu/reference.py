"""Plain-Python reference CPU: the ground truth the Transformer is checked against."""
from __future__ import annotations

from dataclasses import dataclass, replace

from .isa import NMEM, NREG, Instr, Op, Program, clamp_a, wrap


@dataclass(frozen=True)
class State:
    program: tuple[Instr, ...]
    regs: tuple[int, ...]
    pc: int
    halted: bool
    mem: tuple[int, ...]

    def replace(self, **kw) -> "State":
        return replace(self, **kw)


def initial_state(program: Program) -> State:
    return State(program.instrs, (0,) * NREG, 0, False, program.mem)


def step_ref(s: State) -> State:
    if s.halted:
        return s
    ins = s.program[s.pc]
    regs, mem = list(s.regs), list(s.mem)
    pc = clamp_a(s.pc + 1)
    op = ins.op
    if op is Op.HALT:
        return s.replace(halted=True)
    if op is Op.LOADI:
        regs[ins.rd] = wrap(ins.imm)
    elif op is Op.LOAD:
        regs[ins.rd] = mem[clamp_a(regs[ins.ra] + ins.imm)]
    elif op is Op.STORE:
        mem[clamp_a(regs[ins.ra] + ins.imm)] = regs[ins.rb]
    elif op is Op.ADD:
        regs[ins.rd] = wrap(regs[ins.ra] + regs[ins.rb])
    elif op is Op.SUB:
        regs[ins.rd] = wrap(regs[ins.ra] - regs[ins.rb])
    elif op is Op.JZ:
        if regs[ins.ra] == 0:
            pc = clamp_a(ins.imm)
    elif op is Op.JNEG:
        if regs[ins.ra] < 0:
            pc = clamp_a(ins.imm)
    elif op is Op.JMP:
        pc = clamp_a(ins.imm)
    assert len(mem) == NMEM
    return s.replace(regs=tuple(regs), mem=tuple(mem), pc=pc)


def run_ref(program: Program, max_steps: int = 10_000) -> list[State]:
    s = initial_state(program)
    states = [s]
    for _ in range(max_steps):
        s = step_ref(s)
        states.append(s)
        if s.halted:
            break
    return states
