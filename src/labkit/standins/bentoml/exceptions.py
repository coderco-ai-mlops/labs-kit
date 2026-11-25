"""bentoml.exceptions: the errors the practice version raises."""

from bentoml._practice import BentoMLException, NotFound, module_getattr

__all__ = ["BentoMLException", "NotFound"]

__getattr__ = module_getattr("bentoml.exceptions")
