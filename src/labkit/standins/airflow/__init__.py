"""A practice version of Apache Airflow for the labs site, where the real Airflow cannot be installed.

It covers what lab s3-08 uses: a DAG with its settings, PythonOperator tasks, and the links between
tasks made with >> and <<. It does not schedule anything. On your own machine the real Airflow is used.
"""

import importlib.abc
import sys

from airflow._practice import DAG, GO_FURTHER, module_getattr

__version__ = "3.4.0"


class _PlainMessage(importlib.abc.MetaPathFinder):
    """Runs after Python has failed to find an airflow module, to say so in a plain sentence."""

    def find_spec(self, name, path=None, target=None):
        if name.startswith("airflow."):
            raise ImportError(f"The practice version of Airflow on the labs site has no module called `{name}`. {GO_FURTHER}")
        return None


if not any(type(finder).__name__ == "_PlainMessage" for finder in sys.meta_path):
    sys.meta_path.append(_PlainMessage())

from airflow import exceptions, operators, providers, sdk  # noqa: E402  (after the finder, so every part is loaded)

__getattr__ = module_getattr("airflow")
