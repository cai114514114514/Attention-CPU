"""build_cpu(): assemble a hand-wired looped Transformer for one of the architectures.

  v1  six blocks, the original design (spec §4); exact in float64
  v2  two blocks, speculative single-layer execute + two-port memory (spec §7);
      exact in float64 and float32 (float32 also runs on Apple-GPU / MPS)
"""
from __future__ import annotations

import torch

from . import arch_v1, arch_v2
from .layout import Layout
from .nn import LoopedTransformer

ARCHS = {"v1": arch_v1, "v2": arch_v2}
LAYOUTS: dict[str, Layout] = {name: mod.LAYOUT for name, mod in ARCHS.items()}
LAYOUT = arch_v2.LAYOUT          # default architecture


def build_cpu(arch: str = "v2", dtype: torch.dtype = torch.float64, device: str | torch.device = "cpu") -> LoopedTransformer:
    """Build the CPU. Returns an ordinary nn.Module; one forward pass = one machine step."""
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}; choose from {sorted(ARCHS)}")
    if arch == "v1" and dtype != torch.float64:
        raise ValueError("v1 is only exact in float64 (its one-layer wrap-around needs ~2^31 intermediates)")
    mod = ARCHS[arch]
    model = LoopedTransformer([f(mod.LAYOUT) for f in mod.BLOCKS])
    model = model.to(dtype=dtype, device=device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.layout = mod.LAYOUT
    model.arch = arch
    return model
