from pathlib import Path

import pytest

from tfcpu import assemble, run, run_ref
from tfcpu.reference import outputs

EX = Path(__file__).resolve().parent.parent / "examples"


def fib(k):
    a, b = 0, 1
    for _ in range(k):
        a, b = b, a + b
    return a


CASES = {
    "mul": lambda m: m[2] == 42,
    "fib": lambda m: list(m[:20]) == [fib(k) for k in range(20)],
    "bubble_sort": lambda m: list(m[:8]) == sorted([5, -3, 12, 0, 7, 7, -20, 4]),
    "gcd": lambda m: m[2] == 21,
}


@pytest.mark.parametrize("name", CASES)
def test_example_program(model, name):
    prog = assemble((EX / f"{name}.asm").read_text())
    trace = run(model, prog, max_steps=5000)
    ref = run_ref(prog, max_steps=5000)
    assert trace.final.halted
    assert trace.states == ref, "transformer trace diverged from the reference CPU"
    assert CASES[name](trace.final.mem)


PRIMES = [p for p in range(2, 50) if all(p % d for d in range(2, p))]
EXT_CASES = {                                   # use the extended ISA: v2 only
    "factorial": [1, 2, 6, 24, 120, 720, 5040],
    "primes": PRIMES,
}


@pytest.mark.parametrize("name", EXT_CASES)
def test_extended_example_program(model, name):
    prog = assemble((EX / f"{name}.asm").read_text())
    if model.arch == "v1":
        with pytest.raises(ValueError, match="does not implement"):
            run(model, prog)
        return
    trace = run(model, prog, max_steps=50_000)
    ref = run_ref(prog, max_steps=50_000)
    assert trace.final.halted
    assert trace.states == ref, "transformer trace diverged from the reference CPU"
    assert trace.output == outputs(ref) == EXT_CASES[name]
