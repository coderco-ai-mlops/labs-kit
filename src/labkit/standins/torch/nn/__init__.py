"""torch.nn for the practice version of PyTorch: layers, a container and a loss, built on the practice tensors."""

import math
from collections import OrderedDict

import numpy as np

from .._tensor import Tensor, _rng, _wrap, missing, on_cpu
from . import functional

# Real nn.Module methods the practice version leaves out. Any other missing name is a spelling mistake.
_LEFT_OUT = {"add_module", "apply", "buffers", "double", "float", "half", "modules", "named_buffers", "named_children",
             "named_modules", "register_buffer", "register_forward_hook", "register_parameter", "requires_grad_", "type"}


class Parameter(Tensor):
    """A tensor that a model learns: it tracks gradients from the start."""

    def __new__(cls, data, requires_grad=True):
        out = _wrap(data._a, cls)
        out.requires_grad = requires_grad
        return out

    def __init__(self, data, requires_grad=True):
        pass

    def __repr__(self):
        return "Parameter containing:\n" + super().__repr__()


class _AllKeysMatched:
    def __repr__(self):
        return "<All keys matched successfully>"


class Module:
    def __init__(self):
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_modules", OrderedDict())
        self.training = True

    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Module)):
            kind = "parameters" if isinstance(value, Parameter) else "module"
            if "_modules" not in self.__dict__:
                raise AttributeError(f"cannot assign {kind} before Module.__init__() call")
            (self._parameters if kind == "parameters" else self._modules)[name] = value
        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        if name in _LEFT_OUT:
            raise missing(type(self).__name__, name, ())
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f'Module [{type(self).__name__}] is missing the required "forward" function')

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def children(self):
        return iter(self._modules.values())

    def named_parameters(self, prefix=""):
        for name, parameter in self._parameters.items():
            yield prefix + name, parameter
        for name, module in self._modules.items():
            yield from module.named_parameters(prefix + name + ".")

    def parameters(self):
        return (parameter for _, parameter in self.named_parameters())

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.grad = None

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def to(self, *args, **kwargs):
        for where in (*args, *kwargs.values()):
            if not hasattr(where, "np"):   # a dtype is accepted and changes nothing here
                on_cpu(where)
        return self

    def cpu(self):
        return self

    def cuda(self, device="cuda"):
        return self.to(device)

    def state_dict(self):
        return OrderedDict((name, _wrap(parameter._a)) for name, parameter in self.named_parameters())

    def load_state_dict(self, state_dict, strict=True):
        if not hasattr(state_dict, "keys"):
            raise TypeError(f"Expected state_dict to be dict-like, got {type(state_dict)}.")
        own, problems = dict(self.named_parameters()), []
        extra = [f'"{key}"' for key in state_dict if key not in own]
        lost = [f'"{key}"' for key in own if key not in state_dict]
        if strict and lost:
            problems.append(f"Missing key(s) in state_dict: {', '.join(lost)}. ")
        if strict and extra:
            problems.append(f"Unexpected key(s) in state_dict: {', '.join(extra)}. ")
        for key, parameter in own.items():
            if key in state_dict and state_dict[key].shape != parameter.shape:
                problems.append(f"size mismatch for {key}: copying a param with shape {state_dict[key].shape} from checkpoint, "
                                f"the shape in current model is {parameter.shape}.")
        if problems:
            raise RuntimeError(f"Error(s) in loading state_dict for {type(self).__name__}:\n\t" + "\n\t".join(problems))
        for key, parameter in own.items():
            if key in state_dict:
                parameter._a[...] = state_dict[key]._a
        return _AllKeysMatched()

    def extra_repr(self):
        return ""

    def __repr__(self):
        lines = [f"({name}): " + repr(module).replace("\n", "\n  ") for name, module in self._modules.items()]
        inside = "\n  " + "\n  ".join(lines) + "\n" if lines else self.extra_repr()
        return f"{type(self).__name__}({inside})"


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        edge = 1 / math.sqrt(in_features)   # the same range PyTorch draws its starting weights from
        self.weight = Parameter(_wrap(_rng[0].uniform(-edge, edge, (out_features, in_features)).astype(np.float32)))
        self.bias = Parameter(_wrap(_rng[0].uniform(-edge, edge, (out_features,)).astype(np.float32))) if bias else None

    def forward(self, input):
        return functional.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class ReLU(Module):
    def forward(self, input):
        return input.relu()


class Sigmoid(Module):
    def forward(self, input):
        return input.sigmoid()


class Tanh(Module):
    def forward(self, input):
        return input.tanh()


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        for layer in layers:
            self.append(layer)

    def append(self, layer):
        if not isinstance(layer, Module):
            raise TypeError(f"{type(layer).__name__} is not a Module subclass")
        setattr(self, str(len(self._modules)), layer)
        return self

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def __getitem__(self, index):
        layers = list(self._modules.values())
        return Sequential(*layers[index]) if isinstance(index, slice) else layers[index]

    def forward(self, input):
        for layer in self:
            input = layer(input)
        return input


class MSELoss(Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        return functional.mse_loss(input, target, self.reduction)


def __getattr__(name):
    raise missing("torch.nn", name, globals())
