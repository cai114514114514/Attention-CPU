"""Plain-Python reference CPU: the ground truth the Transformer is checked against."""
from __future__ import annotations

from dataclasses import dataclass, replace

from .isa import MUL_STEPS, NMEM, NREG, SP_EMPTY, Instr, Op, Program, clamp_a, clamp_sp, wrap


@dataclass(frozen=True)
class State:
    program: tuple[Instr, ...]
    regs: tuple[int, ...]
    pc: int
    halted: bool
    mem: tuple[int, ...]
    # extended ISA state
    sp: int = SP_EMPTY            # stack pointer, 0..64
    out: int = 0                  # last value written by OUT
    out_flag: bool = False        # True only right after a step that executed OUT
    mstep: int = 0                # MUL micro-step, 0..16
    ma: int = 0                   # MUL multiplicand
    mb: int = 0                   # MUL multiplier (shifted left each micro-step)
    macc: int = 0                 # MUL accumulator

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
    sp, out, out_flag = s.sp, s.out, False
    mstep, ma, mb, macc = s.mstep, s.ma, s.mb, s.macc
    op = ins.op
    if op is Op.HALT:
        return s.replace(halted=True, out_flag=False)
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
    elif op is Op.ADDI:
        regs[ins.rd] = wrap(regs[ins.ra] + ins.imm)
    elif op is Op.JR:
        pc = clamp_a(regs[ins.ra])
    elif op is Op.CALL:
        mem[clamp_a(sp - 1)] = pc                     # return address = clamp(pc + 1)
        sp = clamp_sp(sp - 1)
        pc = clamp_a(ins.imm)
    elif op is Op.RET:
        pc = clamp_a(mem[clamp_a(sp)])
        sp = clamp_sp(sp + 1)
    elif op is Op.PUSH:
        mem[clamp_a(sp - 1)] = regs[ins.rb]
        sp = clamp_sp(sp - 1)
    elif op is Op.POP:
        regs[ins.rd] = mem[clamp_a(sp)]
        sp = clamp_sp(sp + 1)
    elif op is Op.OUT:
        out, out_flag = regs[ins.ra], True
    elif op is Op.ARGMAX:
        regs[ins.rd] = max(range(NMEM), key=lambda j: (mem[j], j))   # largest value, then highest address
    elif op is Op.MUL:
        # multi-cycle: setup, then 16 MSB-first shift-and-add steps; pc holds until the last one
        if mstep == 0:
            ma, mb, macc, mstep = regs[ins.ra], regs[ins.rb], 0, 1
            pc = s.pc
        else:
            macc = wrap(2 * macc + (ma if mb < 0 else 0))
            mb = wrap(2 * mb)
            if mstep == MUL_STEPS:
                regs[ins.rd], mstep = macc, 0
            else:
                mstep, pc = mstep + 1, s.pc
    assert len(mem) == NMEM
    return s.replace(regs=tuple(regs), mem=tuple(mem), pc=pc, sp=sp, out=out, out_flag=out_flag,
                     mstep=mstep, ma=ma, mb=mb, macc=macc)


def run_ref(program: Program, max_steps: int = 10_000) -> list[State]:
    s = initial_state(program)
    states = [s]
    for _ in range(max_steps):
        s = step_ref(s)
        states.append(s)
        if s.halted:
            break
    return states


def outputs(states: list[State]) -> list[int]:
    """The OUT stream of a trace: the value after every step that executed OUT."""
    return [st.out for st in states[1:] if st.out_flag]
