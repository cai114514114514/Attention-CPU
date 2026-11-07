"""Core invariant: model(encode(s)) == encode(step_ref(s)), bit for bit."""
import random

import pytest
import torch

from tfcpu.isa import Op
from tfcpu.reference import step_ref
from tfcpu.state import encode

from strategies import rand_state


@pytest.mark.parametrize("op", list(Op), ids=lambda o: o.name)
def test_one_step_matches_reference(model, op):
    if op not in model.ops:
        pytest.skip(f"{model.arch} does not implement {op.name}")
    rng = random.Random(1000 + int(op))
    states = [rand_state(rng, op_at_pc=op, ops=tuple(sorted(model.ops))) for _ in range(300)]
    L, dt = model.layout, next(model.parameters()).dtype
    x = torch.stack([encode(s, L, dt) for s in states])
    with torch.no_grad():
        y = model(x)
    want = torch.stack([encode(step_ref(s), L, dt) for s in states])
    bad = [i for i in range(len(states)) if not torch.equal(y[i], want[i])]
    assert not bad, f"{len(bad)} mismatches, first state: {states[bad[0]]}"


def test_unbatched_input(model):
    s = rand_state(random.Random(7), op_at_pc=Op.STORE, ops=tuple(sorted(model.ops)))
    L, dt = model.layout, next(model.parameters()).dtype
    with torch.no_grad():
        assert torch.equal(model(encode(s, L, dt)), encode(step_ref(s), L, dt))
