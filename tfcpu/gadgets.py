"""ReLU gadgets (spec §4.1).

An MLP is assembled unit by unit: each hidden unit computes relu(affine(x)) and
adds fixed multiples of its activation into named output fields. Gadgets are
small recipes of such units that compute integer functions exactly.

Every gadget *adds* coef * f(x) to its destination field (residual style).
"""
from __future__ import annotations

from collections import defaultdict

import torch

from .layout import Layout
from .nn import MLP

BIG = float(1 << 20)   # gate constant M: must exceed |x| of anything that is gated


class Aff:
    """An affine expression sum_i c_i * field_i + c over residual fields."""

    __slots__ = ("terms", "c")

    def __init__(self, terms: dict[str, float] | None = None, c: float = 0.0):
        self.terms = dict(terms or {})
        self.c = float(c)

    def _combine(self, other, sign: float) -> "Aff":
        if isinstance(other, (int, float)):
            return Aff(self.terms, self.c + sign * other)
        t = dict(self.terms)
        for k, v in other.terms.items():
            t[k] = t.get(k, 0.0) + sign * v
        return Aff(t, self.c + sign * other.c)

    def __add__(self, other): return self._combine(other, 1.0)
    def __radd__(self, other): return self._combine(other, 1.0)
    def __sub__(self, other): return self._combine(other, -1.0)
    def __rsub__(self, other): return (-self)._combine(other, 1.0)
    def __neg__(self): return self * -1
    def __mul__(self, k: float) -> "Aff": return Aff({n: v * k for n, v in self.terms.items()}, self.c * k)
    __rmul__ = __mul__

    def __repr__(self) -> str:
        parts = [f"{v:+g}*{n}" for n, v in self.terms.items()]
        return " ".join(parts + ([f"{self.c:+g}"] if self.c else [])) or "0"


def F(name: str) -> Aff:
    return Aff({name: 1.0})


class MLPBuilder:
    def __init__(self, layout: Layout):
        self.L = layout
        self.units: list[tuple[Aff, dict[str, float]]] = []
        self.out_bias: dict[str, float] = defaultdict(float)

    def relu(self, x: Aff, out: dict[str, float]) -> None:
        """Add one hidden unit relu(x), feeding out[field] * activation into each field."""
        self.units.append((x, dict(out)))

    def bias(self, dst: str, c: float) -> None:
        self.out_bias[dst] += c

    def build(self) -> MLP:
        L, h = self.L, max(len(self.units), 1)
        mlp = MLP(L.d, h).double()
        W1 = torch.zeros(h, L.d, dtype=torch.float64)
        b1 = torch.zeros(h, dtype=torch.float64)
        W2 = torch.zeros(L.d, h, dtype=torch.float64)
        b2 = torch.zeros(L.d, dtype=torch.float64)
        for j, (x, out) in enumerate(self.units):
            for n, v in x.terms.items():
                W1[j, L[n]] += v
            b1[j] = x.c
            for n, v in out.items():
                W2[L[n], j] += v
        for n, v in self.out_bias.items():
            b2[L[n]] += v
        with torch.no_grad():
            mlp.fc1.weight.copy_(W1); mlp.fc1.bias.copy_(b1)
            mlp.fc2.weight.copy_(W2); mlp.fc2.bias.copy_(b2)
        return mlp


# ---- gadgets: each adds coef * f(x) into dst --------------------------------

def copy(b: MLPBuilder, x: Aff, dst: str, coef: float = 1.0) -> None:
    """x = relu(x) - relu(-x)."""
    b.relu(x, {dst: coef})
    b.relu(-x, {dst: -coef})


def _shift(conds) -> Aff:
    """M * (k - sum s_i): 0 when every binary condition holds, >= M otherwise.

    Subtracting it from a unit's pre-activation switches the unit off unless all
    conditions hold, which is how every gadget below gets an optional `conds`.
    """
    s = Aff()
    for c in conds:
        s = s + c
    return (len(conds) - s) * BIG


def step_ge(b: MLPBuilder, x: Aff, t: int, dst: str, coef: float = 1.0, conds=()) -> None:
    """[x >= t] (and all conds) for integer x."""
    sh = _shift(conds)
    b.relu(x - (t - 1) - sh, {dst: coef})
    b.relu(x - t - sh, {dst: -coef})


def is_zero(b: MLPBuilder, x: Aff, dst: str, coef: float = 1.0, conds=()) -> None:
    """[x == 0] (and all conds) for integer x: the hat relu(x+1) - 2relu(x) + relu(x-1)."""
    sh = _shift(conds)
    b.relu(x + 1 - sh, {dst: coef})
    b.relu(x - sh, {dst: -2 * coef})
    b.relu(x - 1 - sh, {dst: coef})


def is_neg(b: MLPBuilder, x: Aff, dst: str, coef: float = 1.0, conds=()) -> None:
    """[x < 0] (and all conds) for integer x."""
    sh = _shift(conds)
    b.relu(-x - sh, {dst: coef})
    b.relu(-x - 1 - sh, {dst: -coef})


def clamp(b: MLPBuilder, x: Aff, lo: int, hi: int, dst: str, coef: float = 1.0, conds=()) -> None:
    """min(max(x, lo), hi) = lo + relu(x - lo) - relu(x - hi); gated version needs lo == 0."""
    if conds and lo:
        raise ValueError("a gated clamp needs lo == 0 (the constant lo cannot be gated)")
    sh = _shift(conds)
    if lo:
        b.bias(dst, coef * lo)
    b.relu(x - lo - sh, {dst: coef})
    b.relu(x - hi - sh, {dst: -coef})


def wrap16(b: MLPBuilder, x: Aff, dst: str, coef: float = 1.0) -> None:
    """16-bit two's-complement wrap of an integer x in [-65536, 65534], in one layer.

    Exact in float64 only: the step units' activations (~2^15) times 65536 reach
    ~2^31. The float32-exact v2 computes the overflow bits first and scales them
    in the next layer instead.
    """
    copy(b, x, dst, coef)
    step_ge(b, x, 1 << 15, dst, -coef * (1 << 16))
    step_ge(b, -x, (1 << 15) + 1, dst, coef * (1 << 16))   # [x < -32768]


def gate(b: MLPBuilder, x: Aff, conds, dst: str, coef: float = 1.0) -> None:
    """x * prod(conds) for binary conds and |x| < M."""
    sh = _shift(conds)
    b.relu(x - sh, {dst: coef})
    b.relu(-x - sh, {dst: -coef})


def all_of(b: MLPBuilder, conds, dst: str, coef: float = 1.0) -> None:
    """AND of binary conditions: relu(sum s_i - (k - 1))."""
    s = Aff()
    for c in conds:
        s = s + c
    b.relu(s - (len(conds) - 1), {dst: coef})
