"""Runnables: the steps of a chain. Anything with .invoke() that can be joined to the next step with `|`."""

from langchain_core._practice import NotInPracticeVersion, missing, module_getattr

_LEFT_OUT = ("stream", "astream", "astream_events", "batch", "abatch", "ainvoke", "bind", "with_config", "with_retry",
             "with_fallbacks", "with_structured_output", "assign", "pick", "map", "get_graph", "get_prompts",
             "partial", "configurable_fields", "configurable_alternatives", "with_listeners", "as_tool")


def left_out(self, name):
    if name in _LEFT_OUT:
        return NotInPracticeVersion(missing(f".{name}() on a {type(self).__name__}"))
    return AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


def coerce_to_runnable(thing):
    if isinstance(thing, Runnable):
        return thing
    if callable(thing):
        return RunnableLambda(thing)
    if isinstance(thing, dict):
        return RunnableParallel(thing)
    raise TypeError(f"Expected a Runnable, callable or dict.Instead got an unsupported type: {type(thing)}")


class Runnable:
    def __getattr__(self, name):
        raise left_out(self, name)

    def invoke(self, input, config=None, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} needs an invoke method.")

    def __or__(self, other):
        return RunnableSequence(self, coerce_to_runnable(other))

    def __ror__(self, other):
        return RunnableSequence(coerce_to_runnable(other), self)

    def pipe(self, *others):
        return RunnableSequence(self, *(coerce_to_runnable(other) for other in others))


class RunnableSequence(Runnable):
    """Steps run in order. The output of each step is the input of the next. Made by `a | b`."""

    def __init__(self, *steps):
        self.steps = []
        for step in steps:
            step = coerce_to_runnable(step)
            self.steps.extend(step.steps if isinstance(step, RunnableSequence) else [step])
        if len(self.steps) < 2:
            raise ValueError(f"RunnableSequence must have at least 2 steps, got {len(self.steps)}")

    first = property(lambda self: self.steps[0])
    middle = property(lambda self: self.steps[1:-1])
    last = property(lambda self: self.steps[-1])

    def invoke(self, input, config=None, **kwargs):
        for step in self.steps:
            input = step.invoke(input, config)
        return input

    def __repr__(self):
        return "\n| ".join(repr(step) for step in self.steps)


class RunnableLambda(Runnable):
    """A plain function used as a step."""

    def __init__(self, func):
        if not callable(func):
            raise TypeError(f"Expected a callable type for `func`.Instead got an unsupported type: {type(func)}")
        self.func = func

    def invoke(self, input, config=None, **kwargs):
        result = self.func(input)
        return result.invoke(input, config) if isinstance(result, Runnable) else result

    def __repr__(self):
        return f"RunnableLambda({getattr(self.func, '__name__', 'function')})"


class RunnableParallel(Runnable):
    """Gives the same input to several steps and returns a dictionary of their results."""

    def __init__(self, steps=None, **more):
        self.steps__ = {key: coerce_to_runnable(step) for key, step in {**(steps or {}), **more}.items()}

    def invoke(self, input, config=None, **kwargs):
        return {key: step.invoke(input, config) for key, step in self.steps__.items()}

    def __repr__(self):
        return "{\n  " + ",\n  ".join(f"{key}: {step!r}" for key, step in self.steps__.items()) + "\n}"


class RunnablePassthrough(Runnable):
    """Hands its input on unchanged."""

    def invoke(self, input, config=None, **kwargs):
        return input

    def __repr__(self):
        return "RunnablePassthrough()"


__getattr__ = module_getattr("langchain_core.runnables")
