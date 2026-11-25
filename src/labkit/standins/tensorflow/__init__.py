"""A practice version of TensorFlow for the labs site, where the real one cannot be installed.

It covers what lab s4-11 uses: a Keras Sequential model of Dense layers, compile, fit, predict, evaluate,
summary, count_params, get_weights and the random seeds. The sums follow Keras (Glorot starting weights,
the same Adam, SGD and RMSprop steps, float32), so the same code learns the same way on the real TensorFlow.
Everything else stops with a plain sentence. It is never used in a terminal, where the real one is installed.
"""

import difflib
import random as _python_random
import sys
import types
import warnings

import numpy as np

__version__ = "2.21.0"

_rng = np.random.default_rng()
_counts: dict = {}


def _missing(what: str, instead: str = "") -> str:
    return (f"The practice version of TensorFlow on the labs site does not include {what}. "
            "It is in the real TensorFlow: see Go further in this lab." + (f" Here you can use {instead}." if instead else ""))


def _unknown(owner: str, name: str, known) -> str:
    close = difflib.get_close_matches(name, [k for k in known if not k.startswith("_")], 1, 0.8)
    return f"{owner} has nothing called {name}. Did you mean {owner}.{close[0]}?" if close else _missing(f"{owner}.{name}")


def _module(name: str, **members):
    """A module such as tensorflow.keras.layers, so that every style of import works."""
    module = types.ModuleType(name)
    module.__dict__.update(members)
    short = name.replace("tensorflow", "tf", 1)

    def __getattr__(attr):
        if attr.startswith("_"):
            raise AttributeError(f"module '{name}' has no attribute '{attr}'")
        # ImportError, so the sentence also survives `from tensorflow.keras.layers import Conv2D`.
        raise ImportError(_unknown(short, attr, members))

    module.__getattr__ = __getattr__
    sys.modules[name] = module
    return module


class _Plain:
    """Gives a plain sentence for a method or attribute the practice version does not have."""

    _label = "the object"

    def __getattr__(self, name):
        if name.startswith("_") or name in ("optimizer", "loss"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        raise AttributeError(_unknown(self._label, name, dir(type(self)) + list(vars(self))))


def _name(base: str) -> str:
    count = _counts.get(base, 0)
    _counts[base] = count + 1
    return base if count == 0 else f"{base}_{count}"


def set_seed(seed):
    global _rng
    _rng = np.random.default_rng(seed)
    _counts.clear()   # a file that sets the seed first names its layers the same way on every run, as a fresh terminal run does


def set_random_seed(seed):
    _python_random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)


class Tensor(np.ndarray):
    """What a model gives back when it is called directly. `.numpy()` turns it into a plain array."""

    def numpy(self):
        return np.asarray(self)


# Activations. Each one has its slope, written in terms of its own output, for training.

def relu(x):
    return np.maximum(x, 0)


def linear(x):
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


_SLOPES = {relu: lambda a: a > 0, linear: lambda a: 1.0, sigmoid: lambda a: a * (1 - a), tanh: lambda a: 1 - a * a}
_ACTIVATIONS = {fn.__name__: fn for fn in _SLOPES}


class _InputSpec:
    def __init__(self, shape):
        self.shape = (None, *shape)


def Input(shape=None, **kwargs):
    if kwargs:
        raise TypeError(_missing(f"tf.keras.Input({next(iter(kwargs))}=...)"))
    if shape is None:
        raise ValueError("You must pass a `shape` argument.")
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    if len(shape) != 1:
        raise ValueError(_missing(f"an input with the shape {shape}", "one number, as in tf.keras.Input(shape=(3,))"))
    return _InputSpec(shape)


class Layer(_Plain):
    _label = "layer"


class Dense(Layer):
    def __init__(self, units, activation=None, use_bias=True, name=None, **kwargs):
        inputs = kwargs.pop("input_shape", None) or ((kwargs.pop("input_dim"),) if "input_dim" in kwargs else None)
        if kwargs:
            raise TypeError(_missing(f"tf.keras.layers.Dense({next(iter(kwargs))}=...)"))
        if isinstance(units, bool) or not isinstance(units, (int, np.integer)) or units <= 0:
            raise ValueError(f"Received an invalid value for `units`, expected a positive integer. Received: units={units}")
        if activation is None:
            activation = linear
        elif activation in _ACTIVATIONS:
            activation = _ACTIVATIONS[activation]
        elif activation not in _SLOPES:
            if isinstance(activation, str) and not difflib.get_close_matches(activation, _ACTIVATIONS, 1):
                raise ValueError(_missing(f"the activation {activation!r}", "relu, linear, sigmoid and tanh"))
            raise ValueError(f"Could not interpret activation function identifier: {activation}")
        self.units, self.activation, self.use_bias = int(units), activation, use_bias
        self.name = name or _name("dense")
        self.built = False
        self._input_shape = None
        if inputs:
            warnings.warn("Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, "
                          "prefer using an `Input(shape)` object as the first layer in the model instead.", UserWarning, stacklevel=2)
            self._input_shape = tuple(inputs)

    def build(self, input_shape):
        inputs = input_shape if isinstance(input_shape, int) else input_shape[-1]
        limit = np.sqrt(6 / (inputs + self.units))   # Glorot uniform, the Keras default
        self.kernel = _rng.uniform(-limit, limit, (inputs, self.units)).astype("float32")
        self.bias = np.zeros(self.units, dtype="float32") if self.use_bias else None
        self.built = True

    def _variables(self):
        return [self.kernel, self.bias] if self.use_bias else [self.kernel]

    def _need_built(self, action):
        if not self.built:
            raise ValueError(f"You tried to call `{action}` on layer '{self.name}', but the layer isn't built. "
                             "You can build it manually via: `layer.build(input_shape)`.")

    def count_params(self):
        self._need_built("count_params")
        return int(sum(v.size for v in self._variables()))

    def get_weights(self):
        return [v.copy() for v in self._variables()] if self.built else []

    def set_weights(self, weights):
        mine = self._variables() if self.built else []
        if len(weights) != len(mine) or any(np.shape(w) != v.shape for w, v in zip(weights, mine)):
            raise ValueError(f"Layer {self.name} weight shape {[v.shape for v in mine]} is not compatible with "
                             f"provided weight shape {[np.shape(w) for w in weights]}.")
        for variable, new in zip(mine, weights):
            variable[...] = new

    def __call__(self, x):
        x = np.asarray(x, dtype="float32")
        if not self.built:
            self.build(x.shape)
        out = x @ self.kernel
        return self.activation(out + self.bias if self.use_bias else out).view(Tensor)

    def __repr__(self):
        return f"<Dense name={self.name}, built={self.built}>"


# Losses. Each gives the loss as one number and its slope for every prediction.

class _Loss(_Plain):
    _label = "the loss"

    def __init__(self, name=None):
        self.name = name or self._default

    def __call__(self, y_true, y_pred):
        y_true, y_pred = np.asarray(y_true, dtype="float32"), np.asarray(y_pred, dtype="float32")
        return np.float32(self._value(y_pred - y_true))


class MeanSquaredError(_Loss):
    _default = "mean_squared_error"
    _value = staticmethod(lambda diff: np.mean(diff * diff))
    _slope = staticmethod(lambda diff: 2 * diff / diff.size)


class MeanAbsoluteError(_Loss):
    _default = "mean_absolute_error"
    _value = staticmethod(lambda diff: np.mean(np.abs(diff)))
    _slope = staticmethod(lambda diff: np.sign(diff) / diff.size)


_LOSSES = {"mse": MeanSquaredError, "mean_squared_error": MeanSquaredError, "mae": MeanAbsoluteError, "mean_absolute_error": MeanAbsoluteError}


# Optimisers. The update steps are the ones Keras uses.

class Optimizer(_Plain):
    _label = "the optimizer"

    def __init__(self, learning_rate, kwargs):
        if kwargs:
            raise ValueError(f"Argument(s) not recognized: {kwargs}. The size of each step is set with `learning_rate=`.")
        self.learning_rate = float(learning_rate)
        self.iterations = 0
        self._state: dict = {}

    def _apply(self, variables, slopes):
        self.iterations += 1
        for index, (variable, slope) in enumerate(zip(variables, slopes)):
            self._update(index, variable, slope)


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.0, **kwargs):
        super().__init__(learning_rate, kwargs)
        self.momentum = momentum

    def _update(self, index, variable, slope):
        speed = self._state.setdefault(index, np.zeros_like(variable))
        speed[...] = self.momentum * speed - self.learning_rate * slope
        variable += speed


class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.001, rho=0.9, epsilon=1e-7, **kwargs):
        super().__init__(learning_rate, kwargs)
        self.rho, self.epsilon = rho, epsilon

    def _update(self, index, variable, slope):
        average = self._state.setdefault(index, np.zeros_like(variable))
        average[...] = self.rho * average + (1 - self.rho) * slope * slope
        variable -= self.learning_rate * slope / np.sqrt(average + self.epsilon)


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, **kwargs):
        super().__init__(learning_rate, kwargs)
        self.beta_1, self.beta_2, self.epsilon = beta_1, beta_2, epsilon

    def _update(self, index, variable, slope):
        m, v = self._state.setdefault(index, (np.zeros_like(variable), np.zeros_like(variable)))
        m += (slope - m) * (1 - self.beta_1)
        v += (slope * slope - v) * (1 - self.beta_2)
        step = self.learning_rate * np.sqrt(1 - self.beta_2 ** self.iterations) / (1 - self.beta_1 ** self.iterations)
        variable -= step * m / (np.sqrt(v) + self.epsilon)


_OPTIMIZERS = {"adam": Adam, "sgd": SGD, "rmsprop": RMSprop}


class History(_Plain):
    """What fit() gives back. `history.history["loss"]` is the loss after each epoch."""

    _label = "history"

    def __init__(self, losses, params):
        self.history = {"loss": losses}
        self.epoch = list(range(len(losses)))
        self.params = params


def _size(count):
    size = count * 4.0   # every weight is a 32-bit number
    for unit in ("B", "KB", "MB"):
        if size < 1024 or unit == "MB":
            return f"{count:,} ({size:.2f} {unit})"
        size /= 1024


def _bar(steps, tail):
    print(f"{steps}/{steps} {'━' * 20} 0s 1ms/step{tail}")


class Model(_Plain):
    _label = "model"

    def __init__(self, *args, **kwargs):
        raise TypeError(_missing("building a model with tf.keras.Model(...)", "tf.keras.Sequential([...])"))


class Sequential(Model):
    def __init__(self, layers=None, name=None):
        self.name = name or _name("sequential")
        self.layers = []
        self._inputs = None
        for layer in layers or []:
            self.add(layer)

    def add(self, layer):
        if isinstance(layer, _InputSpec):
            if self.layers or self._inputs:
                raise ValueError("tf.keras.Input(...) goes first in the model, and only once.")
            self._inputs = layer.shape[-1]
            return
        if not isinstance(layer, Layer):
            raise TypeError(f"Only instances of `keras.Layer` can be added to a Sequential model. Received: {layer!r} (of type {type(layer)})")
        if not self.layers and self._inputs is None and layer._input_shape:
            self._inputs = layer._input_shape[-1]
        self.layers.append(layer)
        if self._inputs is not None:
            self._build(self._inputs)

    def _build(self, inputs):
        self._inputs = inputs
        for layer in self.layers:
            if not layer.built:
                layer.build(inputs)
            inputs = layer.units

    @property
    def built(self):
        return self._inputs is not None

    @property
    def input_shape(self):
        if not self.built:
            raise AttributeError(f"Sequential model '{self.name}' has no defined input shape yet.")
        return (None, self._inputs)

    @property
    def output_shape(self):
        if not self.built or not self.layers:
            raise AttributeError(f"Sequential model '{self.name}' has no defined output shape yet.")
        return (None, self.layers[-1].units)

    def _data(self, x, what="x", expand=False):
        """Turn what the member passed into a float32 array with one row per example."""
        if not isinstance(x, np.ndarray):
            raise ValueError(f"Unrecognized data type: {what}={x!r} (of type {type(x)}). Pass a NumPy array, for example np.array([[10.0]]).")
        x = np.asarray(x, dtype="float32")
        if x.ndim == 1 and expand:
            x = x.reshape(-1, 1)
        if what != "x":
            return x
        if x.ndim != 2:
            raise ValueError(f"The model needs one row per example, with two pairs of brackets, as in np.array([[10.0]]). It received an input with the shape {x.shape}.")
        if not self.built:
            self._build(x.shape[-1])
        if x.shape[-1] != self._inputs:
            first = self.layers[0].name if self.layers else self.name
            raise ValueError(f"Input 0 of layer '{first}' is incompatible with the layer: expected axis -1 of input shape "
                             f"to have value {self._inputs}, but received input with shape {x.shape}")
        return x

    def _forward(self, x):
        outs = [x]
        for layer in self.layers:
            out = outs[-1] @ layer.kernel
            outs.append(layer.activation(out + layer.bias if layer.use_bias else out))
        return outs

    def __call__(self, x, training=False):
        return self._forward(self._data(np.asarray(x) if isinstance(x, Tensor) else x))[-1].view(Tensor)

    def predict(self, x, batch_size=None, verbose="auto", **kwargs):
        if kwargs:
            raise TypeError(_missing(f"model.predict({next(iter(kwargs))}=...)"))
        x = self._data(x)
        if verbose:
            _bar(max(1, -(-len(x) // (batch_size or 32))), "")
        return np.asarray(self._forward(x)[-1])

    def compile(self, optimizer="rmsprop", loss=None, **kwargs):
        if any(v for v in kwargs.values()):
            raise TypeError(_missing(f"model.compile({next(k for k, v in kwargs.items() if v)}=...)"))
        if isinstance(optimizer, str):
            if optimizer.lower() not in _OPTIMIZERS:
                raise ValueError(f"Could not interpret optimizer identifier: {optimizer}")
            optimizer = _OPTIMIZERS[optimizer.lower()]()
        elif isinstance(optimizer, type) and issubclass(optimizer, Optimizer):
            optimizer = optimizer()
        elif not isinstance(optimizer, Optimizer):
            raise ValueError(f"Could not interpret optimizer identifier: {optimizer}")
        if isinstance(loss, str):
            if loss not in _LOSSES:
                if difflib.get_close_matches(loss, _LOSSES, 1):
                    raise ValueError(f"Could not interpret loss identifier: {loss}")
                raise ValueError(_missing(f"the loss {loss!r}", '"mse" and "mae"'))
            self._loss = _LOSSES[loss]()
        elif isinstance(loss, _Loss) or loss is None:
            self._loss = loss
        else:
            raise ValueError(f"Could not interpret loss identifier: {loss}")
        self.optimizer, self.loss = optimizer, loss

    def _pairs(self, x, y):
        if "optimizer" not in vars(self):
            raise ValueError("You must call `compile()` before using the model.")
        if self._loss is None:
            raise ValueError("No loss to compute. Provide a `loss` argument in `compile()`.")
        x, y = self._data(x, expand=True), self._data(y, "y", expand=True)
        if len(x) != len(y):
            raise ValueError(f"Data cardinality is ambiguous. Make sure all arrays contain the same number of samples. 'x' sizes: {len(x)}, 'y' sizes: {len(y)}")
        return x, y

    def _step(self, x, y):
        """One batch: work out the loss, pass the slopes back through the layers, and let the optimiser move the weights."""
        outs = self._forward(x)
        diff = outs[-1] - y
        delta = self._loss._slope(diff)
        variables, slopes = [], []
        for index in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[index]
            delta = delta * _SLOPES[layer.activation](outs[index + 1])
            variables = layer._variables() + variables
            slopes = ([outs[index].T @ delta, delta.sum(axis=0)] if layer.use_bias else [outs[index].T @ delta]) + slopes
            delta = delta @ layer.kernel.T
        self.optimizer._apply(variables, slopes)
        return float(self._loss._value(diff))

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose="auto", shuffle=True, **kwargs):
        if kwargs:
            raise TypeError(_missing(f"model.fit({next(iter(kwargs))}=...)"))
        x, y = self._pairs(x, y)
        size = batch_size or 32
        steps = max(1, -(-len(x) // size))
        losses = []
        for epoch in range(epochs):
            order = _rng.permutation(len(x)) if shuffle and steps > 1 else np.arange(len(x))
            total = 0.0
            for start in range(0, len(x), size):
                rows = order[start:start + size]
                total += self._step(x[rows], y[rows]) * len(rows)
            losses.append(total / len(x))
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}")
                _bar(steps, f" - loss: {losses[-1]:.4f}")
        return History(losses, {"verbose": verbose, "epochs": epochs, "steps": steps})

    def evaluate(self, x=None, y=None, batch_size=None, verbose="auto", **kwargs):
        if kwargs:
            raise TypeError(_missing(f"model.evaluate({next(iter(kwargs))}=...)"))
        x, y = self._pairs(x, y)
        loss = float(self._loss._value(self._forward(x)[-1] - y))
        if verbose:
            _bar(max(1, -(-len(x) // (batch_size or 32))), f" - loss: {loss:.4f}")
        return loss

    def count_params(self):
        if not self.built:
            raise ValueError(f"You tried to call `count_params` on layer '{self.name}', but the layer isn't built. "
                             "You can build it manually via: `layer.build(input_shape)`.")
        return sum(layer.count_params() for layer in self.layers)

    def get_weights(self):
        return [w for layer in self.layers for w in layer.get_weights()]

    def set_weights(self, weights):
        weights = list(weights)
        for layer in self.layers:
            count = len(layer.get_weights())
            layer.set_weights(weights[:count])
            weights = weights[count:]

    def summary(self):
        widths = (33, 24, 15)

        def line(left, middle, right, fill="─"):
            return left + middle.join(fill * w for w in widths) + right

        def row(edge, name, shape, params):
            return f"{edge} {name:<{widths[0] - 2}} {edge} {shape:<{widths[1] - 2}} {edge} {params:>{widths[2] - 2}} {edge}"

        print(f'Model: "{self.name}"')
        print(line("┏", "┳", "┓", "━"))
        print(row("┃", "Layer (type)", "Output Shape", "Param #"))
        print(line("┡", "╇", "┩", "━"))
        for index, layer in enumerate(self.layers):
            if index:
                print(line("├", "┼", "┤"))
            shape, params = (f"(None, {layer.units})", f"{layer.count_params():,}") if layer.built else ("?", "0 (unbuilt)")
            print(row("│", f"{layer.name} ({type(layer).__name__})", shape, params))
        print(line("└", "┴", "┘"))
        total = self.count_params() if self.built else 0
        print(f" Total params: {_size(total)}")
        print(f" Trainable params: {_size(total)}")
        print(f" Non-trainable params: {_size(0)}")


random = _module("tensorflow.random", set_seed=set_seed)
keras = _module(
    "tensorflow.keras",
    Sequential=Sequential, Model=Model, Input=Input,
    layers=_module("tensorflow.keras.layers", Dense=Dense, Input=Input, Layer=Layer),
    models=_module("tensorflow.keras.models", Sequential=Sequential, Model=Model),
    optimizers=_module("tensorflow.keras.optimizers", Adam=Adam, SGD=SGD, RMSprop=RMSprop, Optimizer=Optimizer),
    losses=_module("tensorflow.keras.losses", MeanSquaredError=MeanSquaredError, MeanAbsoluteError=MeanAbsoluteError),
    activations=_module("tensorflow.keras.activations", **_ACTIVATIONS),
    utils=_module("tensorflow.keras.utils", set_random_seed=set_random_seed),
)


def __getattr__(attr):
    if attr.startswith("_"):
        raise AttributeError(f"module 'tensorflow' has no attribute '{attr}'")
    raise ImportError(_unknown("tf", attr, ("keras", "random")))
