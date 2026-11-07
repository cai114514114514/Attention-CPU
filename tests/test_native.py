"""NativeCPU (C engine) must equal the full v2 model bit for bit, every step."""
import random
from pathlib import Path

import pytest
import torch

from tfcpu import assemble, build_cpu, run_ref
from tfcpu.native import NativeCPU, run_native
from tfcpu.state import encode

from strategies import rand_state

EX = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture(scope="module")
def m32():
    return build_cpu("v2", torch.float32)


@pytest.mark.parametrize("routed", [True, False], ids=["routed", "dense"])
def test_native_equals_full_model_every_step(m32, routed):
    rng = random.Random(777)
    states = [rand_state(rng) for _ in range(300)]
    x = torch.stack([encode(s, m32.layout, torch.float32) for s in states])
    engines = []
    proto = NativeCPU(m32, routed=routed)
    for s in states:
        e = NativeCPU(m32, routed=routed)
        e.load(s)
        engines.append(e)
    with torch.no_grad():
        for t in range(200):
            x = m32(x)
            for i, e in enumerate(engines):
                e.step(1)
                assert torch.equal(e.x, x[i]), f"step {t}, machine {i} differs"


@pytest.mark.parametrize("name", ["mul", "fib", "bubble_sort", "gcd"])
def test_native_examples(m32, name):
    prog = assemble((EX / f"{name}.asm").read_text())
    ref = run_ref(prog, 5000)
    final, steps = run_native(m32, prog, 5000)
    assert final == ref[-1] and steps == len(ref) - 1


def test_native_accepts_float64_model_and_rejects_v1():
    prog = assemble((EX / "gcd.asm").read_text())
    final, _ = run_native(build_cpu("v2"), prog)
    assert final == run_ref(prog)[-1]
    with pytest.raises(ValueError):
        NativeCPU(build_cpu("v1"))


def test_halted_machine_does_not_step(m32):
    prog = assemble("HALT")
    e = NativeCPU(m32)
    e.load(prog)
    assert e.step(10) == 1          # the HALT instruction itself
    assert e.step(10) == 0          # already halted: a fixed point, nothing to do


def test_native_collects_out_stream(m32):
    from tfcpu.reference import outputs
    for name in ("factorial", "primes"):
        prog = assemble((EX / f"{name}.asm").read_text())
        ref = run_ref(prog, 50_000)
        e = NativeCPU(m32)
        e.load(prog)
        assert e.step(50_000) == len(ref) - 1
        assert e.output == outputs(ref) and e.state() == ref[-1]


def test_routing_lists_cover_every_active_unit(m32):
    """Soundness: any unit active on the CPU row must be in its instruction's live list.

    The engines leave the store heads' output on the CPU row at 0 (it is tie-averaged garbage in
    the full model that only feeds scratch clearing; FastCPU/NativeCPU equality tests show it never
    reaches persistent state), so B2 is checked on that same CPU row."""
    from tfcpu.routing import combo_index, route
    from tfcpu.state import CPU
    r = route(m32)
    b1, b2 = m32.blocks
    rng = random.Random(31)
    checked = 0
    for _ in range(2000):
        s = rand_state(rng)
        if s.halted:
            continue
        ins = s.program[s.pc]
        k = combo_index(int(ins.op), ins.rd, ins.ra, ins.rb)
        x = encode(s, m32.layout, torch.float32)
        with torch.no_grad():
            h = x + b1.attn(x)
            act1 = torch.relu(b1.mlp.fc1(h[CPU])) > 0
            h = h + b1.mlp(h)
            h = h + b2.attn(h)
            row = h[CPU].clone()
            row[[m32.layout[f] for f in ("rx_sv", "got_ge", "got_le")]] = 0
            act2 = torch.relu(b2.mlp.fc1(row)) > 0
        assert set(act1.nonzero().flatten().tolist()) <= set(r["b1"][k].tolist())
        assert set(act2.nonzero().flatten().tolist()) <= set(r["b2"][k].tolist())
        checked += 1
    assert checked > 1500


def test_routing_shrinks_work(m32):
    e = NativeCPU(m32)
    assert e.live["b1"].mean() < 0.25 * e.live["h1"] and e.live["b2"].mean() < 0.5 * e.live["h2"]
