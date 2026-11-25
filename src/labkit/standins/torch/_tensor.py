"""Tensors and automatic gradients for the practice version of PyTorch, built on NumPy.

A Tensor wraps a NumPy array. Every operation on tensors that track gradients remembers its inputs
and how to pass a gradient back to each of them, and backward() walks that record in reverse.
"""

import difflib
import functools
import math

import numpy as np

_SITE = "The practice version of PyTorch on the labs site does not "


class NotInPracticeVersion(AttributeError):
    """A PyTorch feature the practice version leaves out. The message is a plain sentence for the member."""


def missing(owner, name, known, kind=NotInPracticeVersion):
    full = f"{owner}.{name}"
    close = difflib.get_close_matches(name, [k for k in known if not k.startswith("_")], n=1, cutoff=0.75)
    if close:
        tail = f"Did you mean {owner}.{close[0]}? If not, it may be in the real PyTorch: see Go further in this lab."
    else:
        tail = "If it is part of the real PyTorch, you can use it there: see Go further in this lab."
    return kind(f"{_SITE}include {full}. {tail}")


def _no_gradient(what):
    return NotInPracticeVersion(f"{_SITE}work out gradients through {what}. The real PyTorch does: see Go further in this lab.")


class DType:
    def __init__(self, name, np_type, label):
        self.name, self.np, self.label = name, np.dtype(np_type), label
        self.is_floating_point = self.np.kind == "f"

    def __repr__(self):
        return "torch." + self.name


float32, float64 = DType("float32", np.float32, "Float"), DType("float64", np.float64, "Double")
int64, int32, bool_ = DType("int64", np.int64, "Long"), DType("int32", np.int32, "Int"), DType("bool", np.bool_, "Bool")
_DTYPES = {d.np: d for d in (float32, float64, int64, int32, bool_)}
_DEFAULT = (float32, int64, bool_)


class Size(tuple):
    def __repr__(self):
        return f"torch.Size({list(self)})"

    def numel(self):
        return math.prod(self)


class device:
    def __init__(self, type="cpu", index=None):
        if isinstance(type, device):
            type, index = type.type, type.index
        name, _, number = str(type).partition(":")
        if name not in ("cpu", "cuda", "mps"):
            raise RuntimeError(f"Expected a device such as 'cpu' or 'cuda'. Got: {type!r}")
        self.type, self.index = name, int(number) if number else index

    def __str__(self):
        return self.type if self.index is None else f"{self.type}:{self.index}"

    def __repr__(self):
        return f"device(type='{self.type}')" if self.index is None else f"device(type='{self.type}', index={self.index})"

    def __eq__(self, other):
        return isinstance(other, device) and (self.type, self.index) == (other.type, other.index)

    def __hash__(self):
        return hash((self.type, self.index))


def on_cpu(where):
    """Everything in the practice version lives on the CPU. Moving to anything else is refused in plain words."""
    if where is not None and device(where).type != "cpu":
        raise RuntimeError(f"There is no GPU on the labs site, so nothing can be moved to '{where}' here. "
                           "Use 'cpu', or choose the device with torch.cuda.is_available() first.")


class GradFn:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"<{self.name} object at {hex(id(self))}>"


_grad_on = [True]
_rng = [np.random.default_rng()]
_LEAF = "a leaf Variable that requires grad is being used in an in-place operation."


class no_grad:
    """Stops gradient tracking, as a `with` block or as a decorator."""

    def __enter__(self):
        self._was, _grad_on[0] = _grad_on[0], False

    def __exit__(self, *exc):
        _grad_on[0] = self._was

    def __call__(self, fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            with no_grad():
                return fn(*args, **kwargs)
        return wrapped


def manual_seed(seed):
    _rng[0] = np.random.default_rng(int(seed))


def _wrap(value, cls=None):
    out = object.__new__(cls or Tensor)
    value = np.asarray(value)
    if value.dtype not in _DTYPES:
        if value.dtype.kind not in "fiub":
            raise TypeError(f"A tensor holds numbers. This data holds something else: {value.tolist()!r}")
        value = value.astype(np.float32 if value.dtype.kind == "f" else np.int64)
    out._a, out.requires_grad, out.grad, out.grad_fn, out._parents, out._freed = value, False, None, None, (), False
    return out


def _op(name, value, *inputs, exact=False):
    """Wrap a NumPy result as a tensor that remembers how to send a gradient back to each input."""
    value = np.asarray(value)
    if not exact and value.dtype == np.float64 and all(t._a.dtype != np.float64 for t, _ in inputs):
        value = value.astype(np.float32)   # PyTorch keeps float32 where NumPy would widen to float64
    elif not exact and value.dtype.kind in "iu" and all(t._a.dtype != value.dtype for t, _ in inputs):
        value = value.astype(np.int64)     # whole numbers are int64 in PyTorch, whatever NumPy's own default is here
    out = _wrap(value)
    tracked = [(t, fn) for t, fn in inputs if t.requires_grad]
    if tracked and _grad_on[0]:
        out.requires_grad, out._parents, out.grad_fn = True, tracked, GradFn(name)
    return out


def _unbroadcast(grad, shape):
    """Sum a gradient back down to the shape of the input it belongs to."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


def _pair(a, b):
    """Turn a number or array on either side of an operator into a tensor of the matching type."""
    if isinstance(a, Tensor) and isinstance(b, Tensor):
        return a, b
    t, other = (a, b) if isinstance(a, Tensor) else (b, a)
    if isinstance(other, np.ndarray):
        other = _wrap(other)
    elif isinstance(other, (bool, int, float, np.number)):
        kind = t._a.dtype
        if kind.kind != "f":
            kind = np.float32 if isinstance(other, (float, np.floating)) else np.int64
        other = _wrap(np.array(other, dtype=kind))
    else:
        return None
    return (a, other) if isinstance(a, Tensor) else (other, b)


def _check_shapes(x, y):
    try:
        np.broadcast_shapes(x.shape, y.shape)
    except ValueError:
        for back in range(1, min(x.ndim, y.ndim) + 1):
            if x.shape[-back] != y.shape[-back] and 1 not in (x.shape[-back], y.shape[-back]):
                raise RuntimeError(f"The size of tensor a ({x.shape[-back]}) must match the size of tensor b ({y.shape[-back]}) "
                                   f"at non-singleton dimension {max(x.ndim, y.ndim) - back}") from None


def _binary(name, f, da=None, db=None):
    """An operator on two tensors. da and db give the gradient for each side. Without them it has no gradient."""
    def op(a, b):
        pair = _pair(a, b)
        if pair is None:
            return NotImplemented
        a, b = pair
        x, y = a._a, b._a
        _check_shapes(x, y)
        with np.errstate(all="ignore"):
            value = f(x, y)
        if da is None:
            if name and _grad_on[0] and (a.requires_grad or b.requires_grad):
                raise _no_gradient(name)
            return _wrap(value)
        return _op(name, value, (a, lambda g: _unbroadcast(da(g, x, y, value), x.shape)),
                   (b, lambda g: _unbroadcast(db(g, x, y, value), y.shape)))
    op.f = f
    return op


_add = _binary("AddBackward0", np.add, lambda g, x, y, v: g, lambda g, x, y, v: g)
_sub = _binary("SubBackward0", np.subtract, lambda g, x, y, v: g, lambda g, x, y, v: -g)
_mul = _binary("MulBackward0", np.multiply, lambda g, x, y, v: g * y, lambda g, x, y, v: g * x)
_div = _binary("DivBackward0", np.true_divide, lambda g, x, y, v: g / y, lambda g, x, y, v: -g * x / (y * y))
_pow = _binary("PowBackward0", np.power, lambda g, x, y, v: g * y * np.power(x, y - 1), lambda g, x, y, v: g * v * np.log(x))


def _swap(op):
    return lambda self, other: op(other, self)


def _inplace(op):
    def method(self, other):
        pair = _pair(self, other)
        if pair is None:
            return NotImplemented
        other = pair[1]
        if _grad_on[0] and (self.requires_grad or other.requires_grad):
            if self.requires_grad and self.grad_fn is None:
                raise RuntimeError(_LEAF)
            old = object.__new__(Tensor)          # the value before the change stays in the record
            old.__dict__.update(self.__dict__)
            self.__dict__.update(op(old, other).__dict__)
            return self
        _check_shapes(self._a, other._a)
        value = op.f(self._a, other._a)
        if not np.can_cast(value.dtype, self._a.dtype, "same_kind"):
            raise RuntimeError(f"result type {_DTYPES[value.dtype].label} can't be cast to the desired output type {self.dtype.label}")
        self._a[...] = value
        return self
    return method


def _unary(name, f, d):
    def method(self):
        x = self._a
        with np.errstate(all="ignore"):
            value = f(x)
        return _op(name, value, (self, lambda g: d(g, x, value)))
    return method


def _view(name, f):
    """An operation that rearranges values. The gradient goes back in the shape the values came from."""
    def method(self, *args):
        x = self._a
        return _op(name, f(x, *args), (self, lambda g: g.reshape(x.shape)), exact=True)
    return method


def _dims(args):
    return tuple(args[0]) if len(args) == 1 and isinstance(args[0], (tuple, list)) else tuple(int(a) for a in args)


def _index(key):
    if isinstance(key, tuple):
        return tuple(_index(k) for k in key)
    return key._a if isinstance(key, Tensor) else key


class Tensor:
    __array_priority__ = 1000

    def __init__(self, data=()):
        self.__dict__.update(tensor(data, dtype=float32).__dict__)

    # What it holds
    shape = property(lambda self: Size(self._a.shape))
    ndim = property(lambda self: self._a.ndim)
    dtype = property(lambda self: _DTYPES[self._a.dtype])
    device = property(lambda self: device("cpu"))
    is_leaf = property(lambda self: self.grad_fn is None)
    data = property(lambda self: _wrap(self._a))
    T = property(lambda self: self.t())

    def size(self, dim=None):
        return self.shape if dim is None else self.shape[dim]

    def dim(self):
        return self._a.ndim

    def numel(self):
        return self._a.size

    def is_floating_point(self):
        return self._a.dtype.kind == "f"

    def item(self):
        if self._a.size != 1:
            raise RuntimeError(f"a Tensor with {self._a.size} elements cannot be converted to Scalar")
        return self._a.reshape(()).item()

    def tolist(self):
        return self._a.tolist()

    def numpy(self):
        if self.requires_grad:
            raise RuntimeError("Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.")
        return self._a

    def __array__(self, dtype=None, copy=None):
        return self.numpy() if dtype is None else self.numpy().astype(dtype)

    def __float__(self):
        return float(self.item())

    def __int__(self):
        return int(self.item())

    def __bool__(self):
        if self._a.size != 1:
            raise RuntimeError("Boolean value of Tensor with more than one value is ambiguous")
        return bool(self.item())

    def __len__(self):
        if self._a.ndim == 0:
            raise TypeError("len() of a 0-d tensor")
        return len(self._a)

    def __iter__(self):
        if self._a.ndim == 0:
            raise TypeError("iteration over a 0-d tensor")
        return (self[i] for i in range(len(self._a)))

    def __format__(self, spec):
        if self._a.ndim == 0:
            return format(self.item(), spec)
        if spec:
            raise TypeError("unsupported format string passed to Tensor.__format__")
        return repr(self)

    def __repr__(self):
        text = "tensor(" + _nest(_cells(self._a), self._a.shape, 7)
        if self.dtype not in _DEFAULT:
            text += f", dtype={self.dtype}"
        if self.grad_fn is not None:
            text += f", grad_fn=<{self.grad_fn.name}>"
        elif self.requires_grad:
            text += ", requires_grad=True"
        return text + ")"

    def __reduce__(self):   # for torch.save and copy.deepcopy: the values, without the gradient record
        return _rebuild, (type(self), self._a.copy(), self.requires_grad and self.grad_fn is None)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        raise missing("Tensor", name, dir(Tensor))

    __hash__ = object.__hash__

    # Type and place
    def _cast(self, kind):
        if kind.np == self._a.dtype:
            return self
        return _op("ToCopyBackward0", self._a.astype(kind.np), (self, lambda g: g), exact=True)

    def to(self, *args, **kwargs):
        out = self
        for arg in (*args, *kwargs.values()):
            if isinstance(arg, DType):
                out = out._cast(arg)
            elif isinstance(arg, (str, device)):
                on_cpu(arg)
        return out

    def cpu(self):
        return self

    def cuda(self, device="cuda"):
        return self.to(device)

    def float(self):
        return self._cast(float32)

    def double(self):
        return self._cast(float64)

    def long(self):
        return self._cast(int64)

    def int(self):
        return self._cast(int32)

    # Gradients
    def requires_grad_(self, requires_grad=True):
        if requires_grad and self._a.dtype.kind != "f":
            raise RuntimeError("only Tensors of floating point dtype can require gradients")
        self.requires_grad = requires_grad
        return self

    def detach(self):
        return _wrap(self._a)

    def clone(self):
        return _op("CloneBackward0", self._a.copy(), (self, lambda g: g), exact=True)

    def backward(self, gradient=None, retain_graph=False):
        if not self.requires_grad:
            raise RuntimeError("element 0 of tensors does not require grad and does not have a grad_fn")
        if gradient is None and self._a.size != 1:
            raise RuntimeError("grad can be implicitly created only for scalar outputs")
        order, seen, stack = [], set(), [(self, False)]
        while stack:   # list every tensor the result was built from, inputs before the results that use them
            node, done = stack.pop()
            if done:
                order.append(node)
            elif id(node) not in seen:
                seen.add(id(node))
                stack.append((node, True))
                stack.extend((parent, False) for parent, _ in node._parents)
        grads = {id(self): np.ones_like(self._a) if gradient is None else gradient._a}
        with np.errstate(all="ignore"):
            for node in reversed(order):
                grad = grads.pop(id(node), None)
                if node._freed:
                    raise RuntimeError("Trying to backward through the graph a second time. The record of the calculation is "
                                       "cleared when you call .backward(). Work out the loss again before you call it again.")
                if grad is None:
                    continue
                if node.grad_fn is None:   # a starting tensor, such as a weight: add to the gradient it already holds
                    grad = grad.astype(node._a.dtype)
                    if node.grad is None:
                        node.grad = _wrap(grad.copy())
                    else:
                        node.grad._a += grad
                    continue
                for parent, fn in node._parents:
                    share = fn(grad)
                    grads[id(parent)] = share if id(parent) not in grads else grads[id(parent)] + share
                if not retain_graph:
                    node._parents, node._freed = (), True

    def _guard(self):
        if _grad_on[0] and self.requires_grad:
            if self.grad_fn is None:
                raise RuntimeError(_LEAF)
            raise _no_gradient("a change made in place to a tensor that is part of a calculation")

    def fill_(self, value):
        self._guard()
        self._a[...] = value
        return self

    def zero_(self):
        return self.fill_(0)

    def __setitem__(self, key, value):
        self._guard()
        self._a[_index(key)] = value._a if isinstance(value, Tensor) else value

    def __getitem__(self, key):
        x, key = self._a, _index(key)

        def back(g):
            full = np.zeros(x.shape, g.dtype)
            np.add.at(full, key, g)
            return full
        return _op("IndexBackward0", x[key], (self, back), exact=True)

    # Maths
    __add__, __radd__, __iadd__ = _add, _swap(_add), _inplace(_add)
    __sub__, __rsub__, __isub__ = _sub, _swap(_sub), _inplace(_sub)
    __mul__, __rmul__, __imul__ = _mul, _swap(_mul), _inplace(_mul)
    __truediv__, __rtruediv__, __itruediv__ = _div, _swap(_div), _inplace(_div)
    __pow__, __rpow__, pow = _pow, _swap(_pow), _pow
    add, sub, mul, div = _add, _sub, _mul, _div
    add_, sub_, mul_, div_ = __iadd__, __isub__, __imul__, __itruediv__
    __floordiv__, __mod__ = _binary("the // operator", np.floor_divide), _binary("the % operator", np.mod)
    __eq__, __ne__ = _binary("", np.equal), _binary("", np.not_equal)
    __lt__, __le__, __gt__, __ge__ = (_binary("", f) for f in (np.less, np.less_equal, np.greater, np.greater_equal))

    __neg__ = _unary("NegBackward0", np.negative, lambda g, x, v: -g)
    abs = __abs__ = _unary("AbsBackward0", np.abs, lambda g, x, v: g * np.sign(x))
    exp = _unary("ExpBackward0", np.exp, lambda g, x, v: g * v)
    log = _unary("LogBackward0", np.log, lambda g, x, v: g / x)
    sqrt = _unary("SqrtBackward0", np.sqrt, lambda g, x, v: g / (2 * v))
    relu = _unary("ReluBackward0", lambda x: np.maximum(x, 0), lambda g, x, v: g * (x > 0))
    sigmoid = _unary("SigmoidBackward0", lambda x: 1 / (1 + np.exp(-x)), lambda g, x, v: g * v * (1 - v))
    tanh = _unary("TanhBackward0", np.tanh, lambda g, x, v: g * (1 - v * v))

    def square(self):
        return self * self

    def _reduce(self, how, dim, keepdim):
        x = self._a
        if how == "mean" and x.dtype.kind != "f":
            raise NotImplementedError("mean(): could not infer output dtype. Input dtype must be either a floating point or "
                               f"complex dtype. Got: {self.dtype.label}")
        value = getattr(x, how)(axis=dim, keepdims=keepdim)
        if value.dtype.kind in "iu":
            value = value.astype(np.int64)
        count = x.size / max(np.size(value), 1) if how == "mean" else 1

        def back(g):
            if dim is not None and not keepdim:
                g = np.expand_dims(g, dim)
            return np.broadcast_to(g, x.shape) / count
        return _op(how.capitalize() + "Backward0", value, (self, back))

    def sum(self, dim=None, keepdim=False):
        return self._reduce("sum", dim, keepdim)

    def mean(self, dim=None, keepdim=False):
        return self._reduce("mean", dim, keepdim)

    def _pick(self, how, dim=None):
        if dim is not None and how != "argmax":
            raise missing("Tensor", f"{how}(dim=...)", ())
        if _grad_on[0] and self.requires_grad and how != "argmax":
            raise _no_gradient(f"Tensor.{how}()")
        value = getattr(self._a, how)(axis=dim)
        return _wrap(value.astype(np.int64) if how == "argmax" else value)

    def max(self, dim=None):
        return self._pick("max", dim)

    def min(self, dim=None):
        return self._pick("min", dim)

    def argmax(self, dim=None):
        return self._pick("argmax", dim)

    def matmul(self, other):
        if not isinstance(other, Tensor):
            return NotImplemented
        x, y = self._a, other._a
        if x.ndim == 0 or y.ndim == 0:
            raise RuntimeError("both arguments to matmul need to be at least 1D")
        rows, cols = (x[None, :] if x.ndim == 1 else x), (y[:, None] if y.ndim == 1 else y)
        if rows.shape[-1] != cols.shape[-2]:
            raise RuntimeError(f"mat1 and mat2 shapes cannot be multiplied ({rows.shape[-2]}x{rows.shape[-1]} and {cols.shape[-2]}x{cols.shape[-1]})")
        if x.dtype != y.dtype:
            raise RuntimeError(f"mat1 and mat2 must have the same dtype, but got {self.dtype.label} and {other.dtype.label}")
        shape = (rows @ cols).shape
        return _op("MmBackward0", x @ y,
                   (self, lambda g: _unbroadcast(g.reshape(shape) @ np.swapaxes(cols, -1, -2), rows.shape).reshape(x.shape)),
                   (other, lambda g: _unbroadcast(np.swapaxes(rows, -1, -2) @ g.reshape(shape), cols.shape).reshape(y.shape)))

    __matmul__ = matmul

    # Shape
    def reshape(self, *shape):
        try:
            return _view("ViewBackward0", lambda x: x.reshape(_dims(shape)))(self)
        except ValueError:
            raise RuntimeError(f"shape '{list(_dims(shape))}' is invalid for input of size {self._a.size}") from None

    view = reshape
    flatten = _view("ViewBackward0", lambda x: x.reshape(-1))
    squeeze = _view("SqueezeBackward0", lambda x, dim=None: x.squeeze(dim) if dim is None or x.shape[dim] == 1 else x)
    unsqueeze = _view("UnsqueezeBackward0", lambda x, dim: np.expand_dims(x, dim))

    def t(self):
        return _op("TBackward0", self._a.T, (self, lambda g: g.T), exact=True)


def _rebuild(cls, array, requires_grad):
    out = _wrap(array, cls)
    out.requires_grad = requires_grad
    return out


def _cells(array):
    """Each value as text, the way PyTorch prints it: `3.` for whole numbers, four decimal places otherwise."""
    flat = array.ravel()
    if array.dtype.kind == "b":
        cells = [str(bool(v)) for v in flat]
    elif array.dtype.kind in "iu":
        cells = [str(int(v)) for v in flat]
    else:
        finite = flat[np.isfinite(flat)]
        sizes = np.abs(finite[finite != 0])
        if (finite == np.round(finite)).all() and (sizes.size == 0 or sizes.max() < 1e8):
            style = "{:.0f}."
        elif sizes.max() / sizes.min() > 1000 or sizes.max() > 1e8 or sizes.min() < 1e-4:
            style = "{:.4e}"
        else:
            style = "{:.4f}"
        cells = [style.format(v) if np.isfinite(v) else str(float(v)) for v in flat]
    width = max((len(c) for c in cells), default=0)
    return [c.rjust(width) for c in cells]


def _nest(cells, shape, indent):
    if not shape:
        return cells[0]
    if len(shape) == 1:   # long rows wrap at 80 characters, as PyTorch's do
        each = max(1, (80 - indent) // (len(cells[0]) + 2)) if cells else 1
        lines = [", ".join(cells[i:i + each]) for i in range(0, len(cells), each)]
        return "[" + (",\n" + " " * (indent + 1)).join(lines) + "]"
    step = len(cells) // shape[0] if shape[0] else 0
    rows = [_nest(cells[i * step:(i + 1) * step], shape[1:], indent + 1) for i in range(shape[0])]
    return "[" + ("," + "\n" * (len(shape) - 1) + " " * (indent + 1)).join(rows) + "]"


# Making tensors

def _rows(data, dim, sizes):
    """PyTorch's own message when the rows of a nested list have different lengths."""
    if isinstance(data, (list, tuple)):
        if dim == len(sizes):
            sizes.append(len(data))
        elif len(data) != sizes[dim]:
            raise ValueError(f"expected sequence of length {sizes[dim]} at dim {dim} (got {len(data)})")
        for value in data:
            _rows(value, dim + 1, sizes)


def _plain(data):
    if isinstance(data, Tensor):
        return data._a
    return [_plain(v) for v in data] if isinstance(data, (list, tuple)) else data


def _made(array, dtype=None, requires_grad=False, device=None):
    on_cpu(device)
    out = _wrap(array if dtype is None else array.astype(dtype.np))
    if requires_grad:
        if out._a.dtype.kind != "f":
            raise RuntimeError("Only Tensors of floating point and complex dtype can require gradients")
        out.requires_grad = True
    return out


def tensor(data, dtype=None, requires_grad=False, device=None):
    if isinstance(data, (Tensor, np.ndarray)):
        array = np.array(_plain(data))
    else:
        _rows(data, 0, [])
        array = np.array(_plain(data))
        if array.dtype.kind in "fiu":
            array = array.astype(np.float32 if array.dtype.kind == "f" else np.int64)
    return _made(array, dtype, requires_grad, device)


def from_numpy(array):
    if not isinstance(array, np.ndarray):
        raise TypeError(f"expected np.ndarray (got {type(array).__name__})")
    return _wrap(array)


def zeros(*size, **options):
    return _made(np.zeros(_dims(size), np.float32), **options)


def ones(*size, **options):
    return _made(np.ones(_dims(size), np.float32), **options)


def rand(*size, **options):
    return _made(_rng[0].random(_dims(size), np.float32), **options)


def randn(*size, **options):
    return _made(_rng[0].standard_normal(_dims(size), np.float32), **options)


def randperm(n):
    return _wrap(_rng[0].permutation(n).astype(np.int64))


def arange(*args, **options):
    array = np.arange(*args)
    return _made(array.astype(np.float32 if array.dtype.kind == "f" else np.int64), **options)


def linspace(start, end, steps, **options):
    return _made(np.linspace(start, end, steps, dtype=np.float32), **options)


def zeros_like(input, **options):
    return _made(np.zeros_like(input._a), **options)


def ones_like(input, **options):
    return _made(np.ones_like(input._a), **options)


def cat(tensors, dim=0):
    arrays = [t._a for t in tensors]
    cuts = np.cumsum([a.shape[dim] for a in arrays])[:-1]
    return _op("CatBackward0", np.concatenate(arrays, axis=dim),
               *[(t, lambda g, i=i: np.split(g, cuts, axis=dim)[i]) for i, t in enumerate(tensors)], exact=True)


def stack(tensors, dim=0):
    return cat([t.unsqueeze(dim) for t in tensors], dim)


def allclose(input, other, rtol=1e-05, atol=1e-08):
    return input.shape == other.shape and bool(np.allclose(input._a, other._a, rtol=rtol, atol=atol))


def equal(input, other):
    return input.shape == other.shape and bool((input._a == other._a).all())


def is_tensor(value):
    return isinstance(value, Tensor)
