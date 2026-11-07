import random

import pytest
import torch

from tfcpu.build import LAYOUT
from tfcpu.state import CPU, DecodeError, decode, encode

from strategies import rand_state


def test_roundtrip():
    rng = random.Random(1)
    for _ in range(50):
        s = rand_state(rng)
        assert decode(encode(s)) == s


def test_decode_rejects_dirty_scratch_and_foreign_fields():
    s = rand_state(random.Random(2))
    x = encode(s)
    y = x.clone(); y[5, LAYOUT["A"]] = 1
    with pytest.raises(DecodeError, match="scratch"):
        decode(y)
    y = x.clone(); y[100, LAYOUT["R2"]] = 3          # a register value on a MEM token
    with pytest.raises(DecodeError, match="canonical"):
        decode(y)
    y = x.clone(); y[CPU, LAYOUT["R0"]] = 0.5
    with pytest.raises(DecodeError, match="integer"):
        decode(y)
    assert torch.equal(encode(decode(x)), x)
