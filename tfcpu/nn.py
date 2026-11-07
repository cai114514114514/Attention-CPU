"""The network: nothing but Linear layers, ReLU, hardmax attention and residual adds."""
from __future__ import annotations

import torch
from torch import nn


def hardmax(scores: torch.Tensor) -> torch.Tensor:
    """Weight 1/k on the k positions sharing the maximal score (averaging hard attention)."""
    w = (scores == scores.amax(dim=-1, keepdim=True)).to(scores.dtype)
    return w / w.sum(dim=-1, keepdim=True)


class HardmaxAttention(nn.Module):
    """Multi-head attention whose softmax is replaced by hardmax.

    Each query puts weight 1/k on the k keys that share the maximal score
    (averaging hard attention). No masking, no positional tricks: positions live
    in the residual stream like any other feature. Heads are laid out
    contiguously in q/k (d_k each) and v (d_v each), as in standard MHA.
    """

    def __init__(self, d_model: int, d_k: int, d_v: int, n_heads: int = 1):
        super().__init__()
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
        self.q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.k = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.v = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.o = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.record = False
        self.last_weights: torch.Tensor | None = None   # [..., heads, T, T]

    def _split(self, t: torch.Tensor, d: int) -> torch.Tensor:
        return t.unflatten(-1, (self.n_heads, d)).transpose(-2, -3)   # [..., H, T, d]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k = self._split(self.q(x), self.d_k), self._split(self.k(x), self.d_k)
        v = self._split(self.v(x), self.d_v)
        w = hardmax(q @ k.transpose(-1, -2))
        if self.record:
            self.last_weights = w.detach()
        out = (w @ v).transpose(-2, -3).flatten(-2)                      # [..., T, H*d_v]
        return self.o(out)


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
