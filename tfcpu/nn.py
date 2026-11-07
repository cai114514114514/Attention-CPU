"""The network: nothing but Linear layers, ReLU, hardmax attention and residual adds."""
from __future__ import annotations

import torch
from torch import nn


class HardmaxAttention(nn.Module):
    """Single-head attention whose softmax is replaced by hardmax.

    Each query puts weight 1/k on the k keys that share the maximal score
    (averaging hard attention). No masking, no positional tricks: positions live
    in the residual stream like any other feature.
    """

    def __init__(self, d_model: int, d_k: int, d_v: int):
        super().__init__()
        self.q = nn.Linear(d_model, d_k, bias=False)
        self.k = nn.Linear(d_model, d_k, bias=False)
        self.v = nn.Linear(d_model, d_v, bias=False)
        self.o = nn.Linear(d_v, d_model, bias=False)
        self.record = False
        self.last_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.q(x) @ self.k(x).transpose(-1, -2)
        w = (scores == scores.amax(dim=-1, keepdim=True)).to(x.dtype)
        w = w / w.sum(dim=-1, keepdim=True)
        if self.record:
            self.last_weights = w.detach()
        return self.o(w @ self.v(x))


class MLP(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(d_hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, attn: HardmaxAttention, mlp: MLP, name: str = ""):
        super().__init__()
        self.attn = attn
        self.mlp = mlp
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class LoopedTransformer(nn.Module):
    """One forward pass = one machine step. Run it in a loop: x = model(x)."""

    def __init__(self, blocks: list[Block]):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x
