from pathlib import Path

import pytest

from tfcpu import assemble, run, run_ref

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
