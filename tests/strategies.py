"""Random machine states and programs for the step / fuzz tests."""
import random

from tfcpu.isa import NMEM, NOPS, NPROG, NREG, WMAX, WMIN, Instr, Op
from tfcpu.reference import State


def rand_word(rng: random.Random) -> int:
    r = rng.random()
    if r < 0.15:
        return rng.choice([WMIN, WMIN + 1, -1, 0, 1, WMAX - 1, WMAX])
    if r < 0.6:
        return rng.randint(-80, 80)
    return rng.randint(WMIN, WMAX)


def rand_instr(rng: random.Random, op: Op | None = None) -> Instr:
    if op is None:
        # HALT is rare so that random programs run for a while
        op = Op(rng.choices(range(NOPS), weights=[1] + [10] * (NOPS - 1))[0])
    rd, ra, rb = (rng.randrange(NREG) for _ in range(3))
    if op in (Op.JZ, Op.JNEG, Op.JMP):
        imm = rng.randrange(NPROG) if rng.random() < 0.85 else rand_word(rng)
    elif op in (Op.LOAD, Op.STORE):
        imm = rng.randint(-5, 70) if rng.random() < 0.8 else rand_word(rng)
    else:
        imm = rand_word(rng)
    return Instr(op, rd, ra, rb, imm)


def rand_state(rng: random.Random, op_at_pc: Op | None = None) -> State:
    prog = [rand_instr(rng) for _ in range(NPROG)]
    pc = rng.randrange(NPROG) if rng.random() < 0.9 else rng.choice([0, NPROG - 1])
    if op_at_pc is not None:
        prog[pc] = rand_instr(rng, op_at_pc)
    regs = tuple(rand_word(rng) for _ in range(NREG))
    mem = tuple(rand_word(rng) for _ in range(NMEM))
    return State(tuple(prog), regs, pc, rng.random() < 0.1, mem)
