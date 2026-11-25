"""torch.nn.functional for the practice version of PyTorch: the same maths as the layers, as plain functions."""

import warnings

from .._tensor import GradFn, Tensor, missing


def _renamed(out, name):
    if out.grad_fn is not None:
        out.grad_fn = GradFn(name)
    return out


def relu(input):
    return input.relu()


def sigmoid(input):
    return input.sigmoid()


def tanh(input):
    return input.tanh()


def linear(input, weight, bias=None):
    if not isinstance(input, Tensor):
        raise TypeError(f"linear(): argument 'input' (position 1) must be Tensor, not {type(input).__name__}")
    out = input @ weight.T
    return _renamed(out if bias is None else out + bias, "AddmmBackward0" if input.ndim == 2 and bias is not None else "AddBackward0")


def mse_loss(input, target, reduction="mean"):
    if input.shape != target.shape:
        warnings.warn(f"Using a target size ({target.shape}) that is different to the input size ({input.shape}). "
                      "This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.",
                      UserWarning, stacklevel=3)
    gap = input - target
    squared = gap * gap
    out = squared if reduction == "none" else squared.sum() if reduction == "sum" else squared.mean()
    return _renamed(out, "MseLossBackward0")

__path__ = []   # so that `from ... import` of a name that is not here reaches the plain sentence too


def __getattr__(name):
    raise missing("torch.nn.functional", name, globals())
