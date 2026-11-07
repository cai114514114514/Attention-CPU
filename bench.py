"""Benchmark: instructions per second for each architecture / precision / device / engine.

    .venv/bin/python bench.py            # prints a markdown table

Every configuration runs bubble sort (random data per machine) and, after timing,
checks a few machines against the reference CPU, so a fast-but-wrong result
cannot show up in the table. "native" (the C engine) runs its machines one after
another on a single CPU core, so its number is single-machine speed.
"""
from __future__ import annotations

import random
import time
from pathlib import Path

import torch

from tfcpu import assemble, build_cpu, decode, encode, initial_state, step_ref
from tfcpu.fast import FastCPU
from tfcpu.native import NativeCPU

SRC = (Path(__file__).parent / "examples" / "bubble_sort.asm").read_text()
DATA = ".data 0  5 -3 12 0 7 7 -20 4"
HAS_MPS = torch.backends.mps.is_available()


def sort_program(rng: random.Random):
    data = " ".join(str(rng.randint(-1000, 1000)) for _ in range(8))
    return assemble(SRC.replace(DATA, f".data 0 {data}"))


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def bench(arch, dtype, device, engine, batch, steps, seed=0):
    rng = random.Random(seed)
    m = build_cpu(arch, dtype, device)
    progs = [sort_program(rng) for _ in range(batch)]
    x0 = torch.stack([encode(initial_state(p), m.layout, dtype) for p in progs]).to(device)
    if engine == "native":                                    # one machine at a time, one CPU core
        eng, dt, finals, done = NativeCPU(m), 0.0, [], 0
        for p in progs:
            eng.load(p)
            t0 = time.perf_counter()
            done += eng.step(steps)                           # fewer if the sort halts early
            dt += time.perf_counter() - t0
            finals.append(eng.state())
        for i in sorted({0, batch // 2, batch - 1}):
            s = initial_state(progs[i])
            for _ in range(steps):
                s = step_ref(s)
            assert finals[i] == s, f"native: machine {i} is wrong"
        return done / dt, dt / done
    if engine == "full":
        m(x0); sync(device)                                   # warm-up
        x = x0
        t0 = time.perf_counter()
        for _ in range(steps):
            x = m(x)
        sync(device)
    else:
        eng = FastCPU(m)
        eng.load(x0); eng.step(); sync(device)                # warm-up
        eng.load(x0)
        t0 = time.perf_counter()
        for _ in range(steps):
            eng.step()
        sync(device)
        x = eng.x
    dt = time.perf_counter() - t0
    for i in sorted({0, batch // 2, batch - 1}):              # correctness spot-check
        s = initial_state(progs[i])
        for _ in range(steps):
            s = step_ref(s)
        assert decode(x[i], m.layout) == s, f"{arch}/{dtype}/{device}/{engine}: machine {i} is wrong"
    return batch * steps / dt, dt / steps / batch


def main():
    f32, f64 = torch.float32, torch.float64
    rows = [
        # (arch, dtype, device, engine, batch, steps)
        ("v1", f64, "cpu", "full", 1, 512),
        ("v2", f64, "cpu", "full", 1, 512),
        ("v2", f32, "cpu", "full", 1, 512),
        ("v2", f32, "mps", "full", 1, 512),
        ("v2", f32, "cpu", "fast", 1, 512),
        ("v2", f32, "mps", "fast", 1, 512),
        ("v2", f32, "cpu", "native", 1, 512),                # C engine; batch rows below run machines one by one
        ("v2", f32, "cpu", "native", 500, 512),
        ("v1", f64, "cpu", "full", 512, 128),
        ("v2", f32, "cpu", "full", 512, 128),
        ("v2", f32, "mps", "full", 512, 128),
        ("v2", f32, "mps", "full", 4096, 128),
        ("v2", f32, "cpu", "fast", 4096, 256),
        ("v2", f32, "mps", "fast", 4096, 256),
        ("v2", f32, "mps", "fast", 16384, 256),
    ]
    print("| arch | dtype | device | engine | machines | instr/s | µs/instr |")
    print("|---|---|---|---|---:|---:|---:|")
    for arch, dt, dev, eng, b, n in rows:
        if dev == "mps" and not HAS_MPS:
            continue
        ips, sps = bench(arch, dt, dev, eng, b, n)
        print(f"| {arch} | {str(dt).removeprefix('torch.')} | {dev} | {eng} | {b:,} | {ips:,.0f} | {sps * 1e6:.2f} |", flush=True)


if __name__ == "__main__":
    main()
