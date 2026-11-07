"""Random machine states and programs for the step / fuzz tests."""
import random

from tfcpu.isa import MUL_STEPS, NMEM, NPROG, NREG, SP_EMPTY, WMAX, WMIN, Instr, Op
from tfcpu.reference import State

ALL_OPS = tuple(Op)


def rand_word(rng: random.Random) -> int:
    r = rng.random()
    if r < 0.15:
        return rng.choice([WMIN, WMIN + 1, -1, 0, 1, WMAX - 1, WMAX])
    if r < 0.6:
        return rng.randint(-80, 80)
    return rng.randint(WMIN, WMAX)


def rand_instr(rng: random.Random, op: Op | None = None, ops=ALL_OPS) -> Instr:
    if op is None:
        # HALT is rare so that random programs run for a while
        op = rng.choices(ops, weights=[1 if o is Op.HALT else 10 for o in ops])[0]
    rd, ra, rb = (rng.randrange(NREG) for _ in range(3))
    if op in (Op.JZ, Op.JNEG, Op.JMP, Op.CALL):
        imm = rng.randrange(NPROG) if rng.random() < 0.85 else rand_word(rng)
    elif op in (Op.LOAD, Op.STORE):
        imm = rng.randint(-5, 70) if rng.random() < 0.8 else rand_word(rng)
    else:
        imm = rand_word(rng)
    return Instr(op, rd, ra, rb, imm)


def rand_state(rng: random.Random, op_at_pc: Op | None = None, ops=ALL_OPS) -> State:
    """A random state. With only base ops, the extended fields stay at their reset values."""
    prog = [rand_instr(rng, ops=ops) for _ in range(NPROG)]
    pc = rng.randrange(NPROG) if rng.random() < 0.9 else rng.choice([0, NPROG - 1])
    if op_at_pc is not None:
        prog[pc] = rand_instr(rng, op_at_pc)
    regs = tuple(rand_word(rng) for _ in range(NREG))
    mem = tuple(rand_word(rng) for _ in range(NMEM))
    s = State(tuple(prog), regs, pc, rng.random() < 0.1, mem)
    if any(o.value >= 9 for o in ops):
        s = s.replace(
            sp=rng.choice([SP_EMPTY, 0, 1, rng.randint(0, SP_EMPTY)]),
            out=rand_word(rng), out_flag=rng.random() < 0.5,
            mstep=0 if rng.random() < 0.5 else rng.randint(1, MUL_STEPS),
            ma=rand_word(rng), mb=rand_word(rng), macc=rand_word(rng),
        )
    return s
