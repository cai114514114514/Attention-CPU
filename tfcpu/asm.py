"""A tiny two-pass assembler for the tfcpu ISA (spec §2.2)."""
from __future__ import annotations

import re

from .isa import NMEM, NPROG, NREG, WMAX, WMIN, Instr, Op, Program


class AsmError(ValueError):
    pass


# operand kinds per mnemonic: r = register, i = immediate (number or label)
_SIG = {
    Op.HALT: "", Op.LOADI: "ri", Op.LOAD: "rri", Op.STORE: "rri",
    Op.ADD: "rrr", Op.SUB: "rrr", Op.JZ: "ri", Op.JNEG: "ri", Op.JMP: "i",
}
_LABEL = re.compile(r"^([A-Za-z_]\w*)\s*:")


def _err(lineno: int, msg: str) -> AsmError:
    return AsmError(f"line {lineno}: {msg}")


def _num(tok: str, lineno: int) -> int:
    try:
        return int(tok, 0)
    except ValueError:
        raise _err(lineno, f"bad number {tok!r}") from None


def assemble(text: str) -> Program:
    labels: dict[str, int] = {}
    lines: list[tuple[int, str, list[str]]] = []   # (lineno, mnemonic, operands)
    mem = [0] * NMEM

    # pass 1: strip comments, collect labels, data, and instruction lines
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.split(";", 1)[0].strip()
        while (m := _LABEL.match(line)):
            name = m.group(1)
            if name in labels:
                raise _err(lineno, f"duplicate label {name!r}")
            labels[name] = len(lines)
            line = line[m.end():].strip()
        if not line:
            continue
        head, _, rest = line.partition(" ")
        if head.lower() == ".data":
            vals = [_num(t, lineno) for t in rest.replace(",", " ").split()]
            if not vals:
                raise _err(lineno, ".data needs an address")
            addr, vals = vals[0], vals[1:]
            if addr < 0 or addr + len(vals) > NMEM:
                raise _err(lineno, f".data {addr}+{len(vals)} words is outside memory 0..{NMEM - 1}")
            for k, v in enumerate(vals):
                if not WMIN <= v <= WMAX:
                    raise _err(lineno, f".data value {v} out of 16-bit range")
                mem[addr + k] = v
            continue
        ops = [t.strip() for t in rest.split(",")] if rest.strip() else []
        lines.append((lineno, head, ops))
        if len(lines) > NPROG:
            raise _err(lineno, f"program longer than {NPROG} instructions")

    # pass 2: encode
    instrs = []
    for lineno, head, ops in lines:
        try:
            op = Op[head.upper()]
        except KeyError:
            raise _err(lineno, f"unknown mnemonic {head!r}") from None
        sig = _SIG[op]
        if len(ops) != len(sig):
            raise _err(lineno, f"{op.name} takes {len(sig)} operand(s), got {len(ops)}")
        vals = []
        for kind, tok in zip(sig, ops):
            if kind == "r":
                m = re.fullmatch(r"[rR](\d+)", tok)
                if not m or int(m.group(1)) >= NREG:
                    raise _err(lineno, f"bad register {tok!r} (use r0..r{NREG - 1})")
                vals.append(int(m.group(1)))
            elif re.fullmatch(r"[A-Za-z_]\w*", tok):
                if tok not in labels:
                    raise _err(lineno, f"undefined label {tok!r}")
                vals.append(labels[tok])
            else:
                v = _num(tok, lineno)
                if not WMIN <= v <= WMAX:
                    raise _err(lineno, f"immediate {v} out of 16-bit range")
                vals.append(v)
        if op is Op.LOADI:
            ins = Instr(op, rd=vals[0], imm=vals[1])
        elif op is Op.LOAD:
            ins = Instr(op, rd=vals[0], ra=vals[1], imm=vals[2])
        elif op is Op.STORE:
            ins = Instr(op, rb=vals[0], ra=vals[1], imm=vals[2])
        elif op in (Op.ADD, Op.SUB):
            ins = Instr(op, rd=vals[0], ra=vals[1], rb=vals[2])
        elif op in (Op.JZ, Op.JNEG):
            ins = Instr(op, ra=vals[0], imm=vals[1])
        elif op is Op.JMP:
            ins = Instr(op, imm=vals[0])
        else:
            ins = Instr(op)
        instrs.append(ins)
    return Program.from_list(instrs, mem)
