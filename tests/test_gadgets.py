"""Every ReLU gadget, exhaustively over its integer input range, must be exact."""
import itertools

import torch

from tfcpu.gadgets import F, MLPBuilder, all_of, clamp, copy, gate, is_neg, is_zero, step_ge, wrap16
from tfcpu.isa import wrap
from tfcpu.layout import Layout

L = Layout(["one", "x", "y", "s1", "s2", "s3", "out"])


def run(build, rows):
    """rows: list of dicts field->value. Returns the 'out' column after x + mlp(x)."""
    b = MLPBuilder(L)
    build(b)
    mlp = b.build()
    x = torch.zeros(len(rows), L.d, dtype=torch.float64)
    for i, r in enumerate(rows):
        x[i, L["one"]] = 1
        for k, v in r.items():
            x[i, L[k]] = v
    with torch.no_grad():
        y = x + mlp(x)
    return y[:, L["out"]].tolist()


def test_copy_negative_and_positive():
    xs = list(range(-70000, 70001, 7))
    out = run(lambda b: copy(b, F("x"), "out", coef=-3), [{"x": v} for v in xs])
    assert out == [-3 * v for v in xs]


def test_step_ge():
    xs = range(-100, 101)
    out = run(lambda b: step_ge(b, F("x"), 5, "out"), [{"x": v} for v in xs])
    assert out == [float(v >= 5) for v in xs]


def test_is_zero_and_is_neg():
    xs = range(-70000, 70001)
    assert run(lambda b: is_zero(b, F("x"), "out"), [{"x": v} for v in xs]) == [float(v == 0) for v in xs]
    assert run(lambda b: is_neg(b, F("x"), "out"), [{"x": v} for v in xs]) == [float(v < 0) for v in xs]


def test_is_zero_gated():
    rows = [{"x": v, "s1": s} for v in range(-300, 301) for s in (0, 1)]
    out = run(lambda b: is_zero(b, F("x") - F("y") * 1, "out", conds=[F("s1")]), rows)
    assert out == [float(r["x"] == 0 and r["s1"] == 1) for r in rows]


def test_clamp():
    xs = range(-65536, 65535)
    out = run(lambda b: clamp(b, F("x") + 1, 0, 63, "out"), [{"x": v} for v in xs])
    assert out == [float(min(max(v + 1, 0), 63)) for v in xs]


def test_wrap16_full_range():
    xs = range(-65536, 65535)
    out = run(lambda b: wrap16(b, F("x"), "out"), [{"x": v} for v in xs])
    assert out == [float(wrap(v)) for v in xs]


def test_wrap16_of_sum():
    rows = [{"x": a, "y": c} for a in range(-32768, 32768, 997) for c in range(-32768, 32768, 1009)]
    out = run(lambda b: wrap16(b, F("x") - F("y"), "out"), rows)
    assert out == [float(wrap(r["x"] - r["y"])) for r in rows]


def test_gate_and_of_three():
    vals = [-65535, -1, 0, 1, 12345, 65535]
    rows = [{"x": v, "s1": a, "s2": c, "s3": d}
            for v in vals for a, c, d in itertools.product((0, 1), repeat=3)]
    out = run(lambda b: gate(b, F("x"), [F("s1"), F("s2"), F("one") - F("s3")], "out"), rows)
    assert out == [float(r["x"] * r["s1"] * r["s2"] * (1 - r["s3"])) for r in rows]


def test_all_of():
    rows = [{"s1": a, "s2": c} for a, c in itertools.product((0, 1), repeat=2)]
    out = run(lambda b: all_of(b, [F("s1"), F("one") - F("s2")], "out"), rows)
    assert out == [float(r["s1"] and not r["s2"]) for r in rows]


def test_output_bias_and_accumulation():
    out = run(lambda b: (copy(b, F("x"), "out"), copy(b, F("y"), "out", coef=2), b.bias("out", -1000)),
              [{"x": 3, "y": 4}])
    assert out == [3 + 8 - 1000]
