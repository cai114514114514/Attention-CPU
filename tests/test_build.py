import pytest
import torch

from tfcpu import build_cpu


def test_v1_refuses_float32():
    with pytest.raises(ValueError, match="float64"):
        build_cpu("v1", torch.float32)


def test_unknown_arch():
    with pytest.raises(ValueError, match="unknown"):
        build_cpu("v9")


def test_v2_has_no_idle_attention():
    m = build_cpu("v2")
    for blk in m.blocks:
        assert (blk.attn.q.weight != 0).any(), f"{blk.name}: attention is all zero"
    assert m.blocks[1].attn.n_heads == 3      # load port + two store comparators


def test_v1_still_six_blocks():
    assert len(build_cpu("v1").blocks) == 6
