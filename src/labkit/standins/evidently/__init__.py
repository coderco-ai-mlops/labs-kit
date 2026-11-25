"""A practice version of Evidently for the labs site, where the real Evidently cannot be installed.

It covers what lab m10 uses, with the API of Evidently 0.5.1: a Report built from the data drift
preset, run on a reference frame and a current frame, and read back as a dictionary, as JSON or as
a small HTML page. The drift scores come from the same statistical tests the real one uses for
numerical columns. On your own machine the real Evidently is used.
"""

import importlib.abc
import sys

from evidently._practice import GO_FURTHER, module_getattr

__version__ = "0.5.1"


class _PlainMessage(importlib.abc.MetaPathFinder):
    """Runs after Python has failed to find an evidently module, to say so in a plain sentence."""

    def find_spec(self, name, path=None, target=None):
        if name.startswith("evidently."):
            raise ImportError(f"The practice version of Evidently on the labs site has no module called `{name}`. {GO_FURTHER}")
        return None


if not any(type(finder).__name__ == "_PlainMessage" and type(finder).__module__ == __name__ for finder in sys.meta_path):
    sys.meta_path.append(_PlainMessage())

from evidently import metric_preset, metrics, report  # noqa: E402  (after the finder, so every part is loaded)

__getattr__ = module_getattr("evidently")
