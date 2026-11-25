"""torch.utils for the practice version of PyTorch."""

from .._tensor import missing
from . import data


def __getattr__(name):
    raise missing("torch.utils", name, globals())
