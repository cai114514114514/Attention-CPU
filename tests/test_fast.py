"""FastCPU must equal the full v2 model bit for bit, every step."""
import random
from pathlib import Path

import pytest
import torch

from tfcpu import assemble, build_cpu, run_ref
from tfcpu.fast import FastCPU, run_fast
from tfcpu.state import encode

from strategies import rand_state

EX = Path(__file__).resolve().parent.parent / "examples"
DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["f64", "f32"])
def test_fast_equals_full_model(dtype, device):
    if device == "mps" and dtype == torch.float64:
        pytest.skip("MPS has no float64")
    m = build_cpu("v2", dtype, device)
    rng = random.Random(4242)
    states = [rand_state(rng) for _ in range(500)]
    x = torch.stack([encode(s, m.layout, dtype) for s in states]).to(device)
    eng = FastCPU(m)
    eng.load(x)
    with torch.no_grad():
        for t in range(300):
            x = m(x)
            eng.step()
            ok = (eng.x == x).flatten(1).all(1)
            assert bool(ok.all()), f"step {t}: {int((~ok).sum())} machines differ"


def test_fast_rejects_v1():
    with pytest.raises(ValueError):
        FastCPU(build_cpu("v1"))


@pytest.mark.parametrize("device", DEVICES)
def test_run_fast_batch_of_examples(device):
    m = build_cpu("v2", torch.float32, device)
    progs = [assemble((EX / f"{n}.asm").read_text()) for n in ("mul", "fib", "bubble_sort", "gcd")]
    finals = run_fast(m, progs, max_steps=5000)
    assert finals == [run_ref(p, 5000)[-1] for p in progs]
    assert run_fast(m, progs[0]) == run_ref(progs[0])[-1]
