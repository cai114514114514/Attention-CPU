"""Run a program on the Transformer. The loop body is only x = model(x)."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .isa import Program
from .nn import LoopedTransformer
from .reference import State, initial_state
from .state import decode, encode


@dataclass
class Trace:
    states: list[State] = field(default_factory=list)

    @property
    def final(self) -> State:
        return self.states[-1]

    @property
    def steps(self) -> int:
        return len(self.states) - 1

    def __len__(self) -> int:
        return len(self.states)


@torch.no_grad()
def run(model: LoopedTransformer, program: Program, max_steps: int = 10_000) -> Trace:
    L, p = model.layout, next(model.parameters())
    x = encode(initial_state(program), L, p.dtype).to(p.device)
    trace = Trace([decode(x, L)])
    for _ in range(max_steps):
        x = model(x)                     # <- the whole computation
        trace.states.append(decode(x, L))  # observation only
        if trace.final.halted:
            break
    return trace
