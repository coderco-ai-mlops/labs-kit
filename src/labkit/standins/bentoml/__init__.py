"""A practice version of BentoML for the labs site, where the real BentoML cannot be installed.

It covers what lab m08b uses: @bentoml.service on a class, @bentoml.api on its methods, a model the
service declares with bentoml.models.BentoModel, and the model store behind bentoml.sklearn.save_model,
bentoml.sklearn.load_model and bentoml.models.get. Calling the service class builds the service, and
calling an API method on it runs the method. It does not serve over HTTP, build a Bento or make a
container image. On your own machine the real BentoML is used.
"""

import importlib.abc
import sys

from bentoml._practice import GO_FURTHER, Model, Tag, api, module_getattr, service

__version__ = "1.4.39"


class _PlainMessage(importlib.abc.MetaPathFinder):
    """Runs after Python has failed to find a bentoml module, to say so in a plain sentence."""

    def find_spec(self, name, path=None, target=None):
        if name.startswith("bentoml."):
            raise ImportError(f"The practice version of BentoML on the labs site has no module called `{name}`. {GO_FURTHER}")
        return None


if not any(type(finder).__name__ == "_PlainMessage" and finder.__module__ == __name__ for finder in sys.meta_path):
    sys.meta_path.append(_PlainMessage())

from bentoml import exceptions, models, sklearn  # noqa: E402  (after the finder, so every part is loaded)

__all__ = ["Model", "Tag", "api", "exceptions", "models", "service", "sklearn"]

__getattr__ = module_getattr("bentoml")
