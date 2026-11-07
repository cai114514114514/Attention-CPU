"""Fast executor for v2: the same weights and the same arithmetic, on fewer rows.

The full model recomputes all 129 token rows every step, but in v2 only the CPU
row and at most one MEM row (the store target) can change. This engine keeps
the attention keys/values of the unchanging rows cached, like an instruction
cache, and per step evaluates:

  B1   the CPU query against the cached PROG keys, then the B1 MLP on the CPU row
  B2   read heads (load, ARGMAX): CPU queries against cached MEM keys/values,
               refreshed for the one row a store changes
       store heads: the 64 MEM queries against the CPU's key only; each MEM query's
               best score among the other (constant) keys is cached, so the CPU wins
               a head iff it beats that score
       B2 MLP on the CPU row, and on the MEM rows where the CPU wins every store
               head (the val write is gated by the AND of all store heads)

Rows' scratch "garbage" from the full model is never needed: every gate that
touches persistent state reads only fields this engine reproduces exactly, and
B2 zeroes scratch either way. tests/test_fast.py checks the result equals the
full model bit for bit on thousands of random programs.
"""
from __future__ import annotations

import torch

from .isa import NMEM, Program
from .nn import LoopedTransformer, hardmax
from .reference import State, initial_state
from .state import CPU, MEM0, decode, encode


class FastCPU:
    def __init__(self, model: LoopedTransformer):
        if getattr(model, "arch", None) != "v2":
            raise ValueError("FastCPU needs the v2 architecture")
        self.model = model
        self.b1, self.b2 = model.blocks
        a2 = self.b2.attn
        self.R = getattr(a2, "n_read_heads", 1)                         # CPU-query heads come first
        if self.b1.attn.n_heads != 1 or a2.n_heads <= self.R:
            raise ValueError("unexpected v2 head layout")
        self.H, self.dk2, self.dv2 = a2.n_heads, a2.d_k, a2.d_v
        self.x: torch.Tensor | None = None

    def _h(self, t: torch.Tensor, d: int) -> torch.Tensor:
        """[..., H*d] -> [..., H, d]"""
        return t.unflatten(-1, (self.H, d))

    # -- cache ------------------------------------------------------------------
    @torch.no_grad()
    def load(self, x: torch.Tensor) -> None:
        """Start from canonical encodings x [B, 129, d]; caches everything that never changes."""
        if x.dim() == 2:
            x = x.unsqueeze(0)
        self.x = x.clone()
        a1, a2, dk, dv, R = self.b1.attn, self.b2.attn, self.dk2, self.dv2, self.R
        self.K1, self.V1 = a1.k(x), a1.v(x)                            # fetch port: PROG keys / fields
        k2, v2 = self._h(a2.k(x), dk), self._h(a2.v(x), dv)            # [B, T, H, d]
        self.K2read = k2[:, :, :R].clone()                              # read heads: keys/values of MEM rows
        self.V2read = v2[:, :, :R].clone()                              #   change only on stores (updated then)
        # store heads: per MEM query, the best score / averaged value over the non-CPU rows
        qm = self._h(a2.q(x[:, MEM0:MEM0 + NMEM]), dk)[:, :, R:]       # [B, 64, S, dk]
        s_other = torch.einsum("bisk,btsk->bist", qm, k2[:, 1:, R:])    # store keys of rows 1..128 never change
        self.Qstore = qm
        self.best_other = s_other.amax(-1)                              # [B, 64, S]
        self.v_other = torch.einsum("bist,btsv->bisv", hardmax(s_other), v2[:, 1:, R:])

    # -- one step ---------------------------------------------------------------
    @torch.no_grad()
    def step(self) -> None:
        x, b1, b2 = self.x, self.b1, self.b2
        a1, a2, dk, dv, R = b1.attn, b2.attn, self.dk2, self.dv2, self.R
        c = x[:, CPU]                                                   # [B, d]
        # B1: fetch + speculative execute (CPU row only)
        w1 = hardmax(torch.einsum("bk,btk->bt", a1.q(c), self.K1))
        c = c + a1.o(torch.einsum("bt,btv->bv", w1, self.V1))
        c1 = c + b1.mlp(c)
        # B2 read heads on the CPU row. The store heads' output on the CPU row is irrelevant
        # (the val gate needs is_mem, and scratch is cleared), so it is left at zero.
        q2 = self._h(a2.q(c1), dk)[:, :R]                               # [B, R, dk]
        w2 = hardmax(torch.einsum("bhk,bthk->bht", q2, self.K2read))
        heads = torch.zeros(c1.shape[0], self.H, dv, dtype=c1.dtype, device=c1.device)
        heads[:, :R] = torch.einsum("bht,bthv->bhv", w2, self.V2read)
        c2 = c1 + a2.o(heads.flatten(-2))
        new_cpu = c2 + b2.mlp(c2)
        # store heads: does the CPU's key beat each MEM query's best other key?
        k_cpu = self._h(a2.k(c1), dk)[:, R:]                            # [B, S, dk]
        s_cpu = torch.einsum("bisk,bsk->bis", self.Qstore, k_cpu)       # [B, 64, S]
        if bool((s_cpu == self.best_other).any()):
            raise RuntimeError("store-port tie: v2 invariant violated")
        win = (s_cpu > self.best_other).all(-1)                         # [B, 64]
        x[:, CPU] = new_cpu
        bi, mi = win.nonzero(as_tuple=True)
        if bi.numel():
            v_cpu = self._h(a2.v(c1), dv)[bi, R:]                       # [n, S, dv]
            hv = torch.zeros(bi.numel(), self.H, dv, dtype=c1.dtype, device=c1.device)
            hv[:, R:] = v_cpu                                           # every store head picked the CPU
            r = x[bi, MEM0 + mi]
            r = r + a2.o(hv.flatten(-2))
            r = r + b2.mlp(r)
            x[bi, MEM0 + mi] = r
            self.K2read[bi, MEM0 + mi] = self._h(a2.k(r), dk)[:, :R]   # refresh read-head caches
            self.V2read[bi, MEM0 + mi] = self._h(a2.v(r), dv)[:, :R]

    @torch.no_grad()
    def run(self, steps: int, check_every: int = 64) -> torch.Tensor:
        """Run up to `steps` steps; stops early once every machine has halted."""
        L = self.model.layout
        for t in range(steps):
            self.step()
            if (t + 1) % check_every == 0 and bool((self.x[:, CPU, L["halted"]] == 1).all()):
                break
        return self.x


@torch.no_grad()
def run_fast(model: LoopedTransformer, programs: list[Program] | Program, max_steps: int = 10_000) -> list[State] | State:
    """Run one program or a batch of programs with FastCPU; returns the final state(s)."""
    single = isinstance(programs, Program)
    progs = [programs] if single else list(programs)
    L, p = model.layout, next(model.parameters())
    x = torch.stack([encode(initial_state(pr), L, p.dtype) for pr in progs]).to(p.device)
    eng = FastCPU(model)
    eng.load(x)
    out = eng.run(max_steps)
    finals = [decode(out[i], L) for i in range(len(progs))]
    return finals[0] if single else finals
