import pytest
import torch

from tfcpu import build_cpu

# Every model-level test runs on each exact configuration.
CONFIGS = [("v1", torch.float64), ("v2", torch.float64), ("v2", torch.float32)]


@pytest.fixture(scope="session", params=CONFIGS, ids=lambda c: f"{c[0]}-{str(c[1]).removeprefix('torch.')}")
def model(request):
    return build_cpu(*request.param)
