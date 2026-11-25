"""Practice version of PyTorch for the labs site.

The real PyTorch cannot be installed inside a browser, so the labs site loads this small version
instead. It is built on NumPy and behaves the same as the real one for everything the PyTorch
labs use: tensors, automatic gradients, layers, losses, optimisers, data loaders, devices and
saving. Anything it leaves out fails with a plain sentence that says so. It is never used in a
terminal, where the real PyTorch is installed.
"""

import pickle as _pickle
import sys as _sys
import types as _types

from . import _tensor, nn, optim, utils
from ._tensor import (NotInPracticeVersion, Size, Tensor, allclose, arange, cat, device, equal, from_numpy, is_tensor,
                      linspace, manual_seed, no_grad, ones, ones_like, rand, randn, randperm, stack, tensor, zeros, zeros_like)
from ._tensor import DType as dtype
from ._tensor import missing as _missing

__version__ = "practice version"


def save(obj, f):
    """Write a state_dict (or anything else) to a file."""
    if hasattr(f, "write"):
        return _pickle.dump(obj, f)
    with open(f, "wb") as handle:
        _pickle.dump(obj, handle)


def load(f, map_location=None, weights_only=True):
    """Read back what torch.save wrote. Like the real PyTorch, it expects weights and refuses a whole model."""
    if hasattr(f, "read"):
        obj = _pickle.load(f)
    else:
        with open(f, "rb") as handle:
            obj = _pickle.load(handle)
    if weights_only and isinstance(obj, nn.Module):
        raise _pickle.UnpicklingError("Weights only load failed. This file holds a whole model, not its weights. "
                                      "Save the weights with torch.save(model.state_dict(), path).")
    return obj


def _part(name, **contents):
    """A small part of torch, such as torch.cuda, that the checks can read and change."""
    part = _types.ModuleType(name)
    part.__dict__.update(contents, __path__=[])
    part.__getattr__ = lambda missing_name: _raise(_missing(name, missing_name, contents))
    _sys.modules[name] = part
    return part


def _raise(error):
    raise error


# The labs site has no GPU, so both of these say no. Code that asks first runs on the CPU.
cuda = _part("torch.cuda", is_available=lambda: False, device_count=lambda: 0)
backends = _part("torch.backends", mps=_part("torch.backends.mps", is_available=lambda: False, is_built=lambda: False))


class _NotIncludedFinder:
    """Gives `import torch.something` the same plain sentence as `torch.something`."""

    @staticmethod
    def find_spec(name, path=None, target=None):
        if name.startswith("torch."):
            owner, _, last = name.rpartition(".")
            known = vars(_sys.modules[owner]) if owner in _sys.modules else ()
            raise _missing(owner, last, known, type("NotInPracticeVersion", (ImportError,), {}))


_sys.meta_path.append(_NotIncludedFinder)

# Type names, and functions that do the same as the method of the same name: torch.sum(x) is x.sum().
globals().update(float32=_tensor.float32, float64=_tensor.float64, int64=_tensor.int64, int32=_tensor.int32,
                 float=_tensor.float32, double=_tensor.float64, long=_tensor.int64, int=_tensor.int32, bool=_tensor.bool_)
for _name in ("sum", "mean", "abs", "exp", "log", "sqrt", "relu", "sigmoid", "tanh", "square", "pow", "matmul", "max", "min",
              "argmax", "reshape", "flatten", "squeeze", "unsqueeze", "add", "sub", "mul", "div", "clone", "numel", "t"):
    globals()[_name] = (lambda method: lambda input, *args, **kwargs: getattr(input, method)(*args, **kwargs))(_name)


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(f"module 'torch' has no attribute '{name}'")
    raise _missing("torch", name, globals())
