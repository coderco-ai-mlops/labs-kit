"""airflow.sdk: where Airflow 3 keeps the names you use to write a DAG file."""

from airflow._practice import DAG, BaseOperator, chain, module_getattr

__getattr__ = module_getattr("airflow.sdk")
