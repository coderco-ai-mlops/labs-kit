"""bentoml.models: the model store."""

from bentoml._practice import BentoModel, Model, module_getattr
from bentoml._practice import every as _every
from bentoml._practice import get as _get

__all__ = ["BentoModel", "get", "list"]


def get(tag) -> Model:
    """The stored model with this tag. "name:latest", or the name alone, gives the newest version."""
    return _get(tag)


def list() -> "list[Model]":  # noqa: A001  (the real function has this name)
    """Every model in the store."""
    return _every()


__getattr__ = module_getattr("bentoml.models")
