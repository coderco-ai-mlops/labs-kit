import warnings

from airflow import _practice
from airflow._practice import AirflowException, DuplicateTaskIdFound, TaskNotFound


class DeprecatedImportWarning(FutureWarning):
    pass


def __getattr__(name: str):
    # The Airflow 2 location of the exceptions a task raises. It still works in Airflow 3, with a warning.
    if name in ("AirflowSkipException", "AirflowFailException"):
        warnings.warn(f"airflow.exceptions.{name} is deprecated and will be removed in a future version. "
                      f"Use airflow.sdk.exceptions.{name} instead.", DeprecatedImportWarning, stacklevel=2)
        return getattr(_practice, name)
    return _practice.module_getattr("airflow.exceptions")(name)
