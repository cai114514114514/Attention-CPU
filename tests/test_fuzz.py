"""Differential fuzzing: random programs from random states, every step compared exactly."""
import random

import torch

from tfcpu.isa import NMEM, NREG
from tfcpu.reference import step_ref
from tfcpu.state import CPU, MEM0, encode

from strategies import rand_state

N_PROGRAMS = 2000
N_STEPS = 300
CHUNK = 500


def expected(base: torch.Tensor, states, L) -> torch.Tensor:
    """encode() for a batch whose programs equal those in `base`: only CPU/MEM values change."""
    want = base.clone()
    want[:, CPU, [L[f"R{i}"] for i in range(NREG)]] = torch.tensor([s.regs for s in states], dtype=base.dtype)
    want[:, CPU, L["pc"]] = torch.tensor([s.pc for s in states], dtype=base.dtype)
    want[:, CPU, L["halted"]] = torch.tensor([float(s.halted) for s in states], dtype=base.dtype)
    want[:, MEM0:MEM0 + NMEM, L["val"]] = torch.tensor([s.mem for s in states], dtype=base.dtype)
    for f in ("sp", "out", "out_flag", "mstep", "ma", "mb", "macc"):
        want[:, CPU, L[f]] = torch.tensor([float(getattr(s, f)) for s in states], dtype=base.dtype)
    return want


def test_random_programs(model):
    rng = random.Random(20261107)
    total_steps = executed = 0
    for _ in range(N_PROGRAMS // CHUNK):
        states = [rand_state(rng, ops=tuple(sorted(model.ops))) for _ in range(CHUNK)]
        L, dt = model.layout, next(model.parameters()).dtype
        x = torch.stack([encode(s, L, dt) for s in states])
        base = x.clone()
        with torch.no_grad():
            for t in range(N_STEPS):
                executed += sum(not s.halted for s in states)
                x = model(x)
                states = [step_ref(s) for s in states]
                want = expected(base, states, L)
                ok = (x == want).flatten(1).all(1)
                assert ok.all(), f"step {t}: {int((~ok).sum())} programs diverged, e.g. #{int((~ok).nonzero()[0])}"
                total_steps += CHUNK
    assert total_steps == N_PROGRAMS * N_STEPS
    assert executed > N_PROGRAMS * N_STEPS // 4, f"too few live steps ({executed}); fuzz is not exercising much"
