"""Residual-stream layout: every dimension has a name (spec §3.2)."""
from __future__ import annotations

from collections.abc import Iterable


class Layout:
    def __init__(self, persistent: Iterable[str], scratch: Iterable[str] = ()):
        self.persistent = list(persistent)
        self.scratch = list(scratch)
        self.names = self.persistent + self.scratch
        self.idx = {n: i for i, n in enumerate(self.names)}
        if len(self.idx) != len(self.names):
            raise ValueError("duplicate field names")

    @property
    def d(self) -> int:
        return len(self.names)

    def __getitem__(self, name: str) -> int:
        return self.idx[name]

    def __contains__(self, name: str) -> bool:
        return name in self.idx

    def __repr__(self) -> str:
        return f"Layout(d={self.d}, persistent={len(self.persistent)}, scratch={len(self.scratch)})"
