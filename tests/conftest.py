import pytest

from tfcpu import build_cpu


@pytest.fixture(scope="session")
def model():
    return build_cpu()
