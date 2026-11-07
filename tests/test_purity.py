"""The model is a plain Transformer and the driver loop does no computation."""
import inspect
import re

from torch import nn

from tfcpu.run import run as run_fn
from tfcpu.nn import MLP, Block, HardmaxAttention, LoopedTransformer

ALLOWED = (LoopedTransformer, nn.ModuleList, Block, HardmaxAttention, MLP, nn.Linear, nn.ReLU)


def test_only_transformer_parts(model):
    for name, m in model.named_modules():
        assert type(m) in ALLOWED, f"{name}: {type(m).__name__} is not a Transformer component"
    for name, p in model.named_parameters():
        owner = model.get_submodule(name.rsplit(".", 1)[0])
        assert isinstance(owner, nn.Linear), f"{name} lives outside a Linear layer"
    assert len(model.blocks) == 6


def test_driver_loop_only_feeds_back():
    src = inspect.getsource(run_fn)
    writes = re.findall(r"^\s*x\s*=\s*(.+?)\s*(?:#.*)?$", src, re.M)
    assert writes == ["encode(initial_state(program))", "model(x)"]
