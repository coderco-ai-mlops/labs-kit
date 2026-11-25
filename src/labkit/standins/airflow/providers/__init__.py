from airflow._practice import module_getattr
from airflow.providers import standard

__getattr__ = module_getattr("airflow.providers")
