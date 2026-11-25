"""A practice version of Feast for the labs site, where the real Feast cannot be installed.

It covers what lab m06 uses, with the API of Feast 0.49: an Entity, a FileSource that points at a
parquet file, a FeatureView with a schema and a ttl, and a FeatureStore with a local provider that
can apply definitions, materialise into an online store, and read features two ways: a point-in-time
join for training and a single-key lookup for inference. On your own machine the real Feast is used.
"""

import importlib.abc
import sys

from feast._practice import GO_FURTHER, Entity, FeatureStore, FeatureView, Field, FileSource, ValueType, module_getattr

__version__ = "0.49.0"


class _PlainMessage(importlib.abc.MetaPathFinder):
    """Runs after Python has failed to find a feast module, to say so in a plain sentence."""

    def find_spec(self, name, path=None, target=None):
        if name.startswith("feast."):
            raise ImportError(f"The practice version of Feast on the labs site has no module called `{name}`. {GO_FURTHER}")
        return None


if not any(type(finder).__name__ == "_PlainMessage" and type(finder).__module__ == __name__ for finder in sys.meta_path):
    sys.meta_path.append(_PlainMessage())

from feast import errors, repo_config, types, value_type  # noqa: E402  (after the finder, so every part is loaded)

__getattr__ = module_getattr("feast")
