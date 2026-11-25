"""The Airflow 2 location of PythonOperator. It still works in Airflow 3, with a warning."""

import warnings

from airflow import _practice


class DeprecatedImportWarning(FutureWarning):
    pass


def __getattr__(name: str):
    if name == "PythonOperator":
        warnings.warn("The `airflow.operators.python.PythonOperator` attribute is deprecated. Please use "
                      "`'airflow.providers.standard.operators.python.PythonOperator'`.", DeprecatedImportWarning, stacklevel=2)
        return _practice.PythonOperator
    return _practice.module_getattr("airflow.operators.python")(name)
