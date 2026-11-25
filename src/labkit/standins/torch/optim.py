"""torch.optim for the practice version of PyTorch: SGD and Adam."""

import numpy as np

from ._tensor import Tensor, missing


class Optimizer:
    def __init__(self, params, **settings):
        params = list(params)
        if not params:
            raise ValueError("optimizer got an empty parameter list")
        for value in params:
            if not isinstance(value, Tensor):
                raise TypeError(f"optimizer can only optimize Tensors, but one of the params is {type(value).__name__}")
        if settings["lr"] < 0:
            raise ValueError(f"Invalid learning rate: {settings['lr']}")
        self.defaults = settings
        self.param_groups = [dict(params=params, **settings)]
        self._memory = {}   # what the optimiser remembers about each weight between steps

    def zero_grad(self, set_to_none=True):
        for group in self.param_groups:
            for weight in group["params"]:
                if set_to_none:
                    weight.grad = None
                elif weight.grad is not None:
                    weight.grad._a[...] = 0

    def step(self):
        for group in self.param_groups:
            for weight in group["params"]:
                if weight.grad is not None:
                    slope = weight.grad._a + group["weight_decay"] * weight._a
                    weight._a -= self._move(id(weight), slope, group).astype(weight._a.dtype)

    def __repr__(self):
        lines = [f"    {key}: {value}" for key, value in self.param_groups[0].items() if key != "params"]
        return f"{type(self).__name__} (\nParameter Group 0\n" + "\n".join(lines) + "\n)"


class SGD(Optimizer):
    def __init__(self, params, lr=0.001, momentum=0, weight_decay=0):
        super().__init__(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    def _move(self, key, slope, group):
        if group["momentum"]:
            slope = self._memory[key] = group["momentum"] * self._memory[key] + slope if key in self._memory else slope
        return group["lr"] * slope


class Adam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    def _move(self, key, slope, group):
        (b1, b2), (average, spread, steps) = group["betas"], self._memory.get(key, (0.0, 0.0, 0))
        average, spread, steps = b1 * average + (1 - b1) * slope, b2 * spread + (1 - b2) * slope * slope, steps + 1
        self._memory[key] = (average, spread, steps)
        return group["lr"] * (average / (1 - b1 ** steps)) / (np.sqrt(spread / (1 - b2 ** steps)) + group["eps"])

__path__ = []   # so that `from ... import` of a name that is not here reaches the plain sentence too


def __getattr__(name):
    raise missing("torch.optim", name, globals())
