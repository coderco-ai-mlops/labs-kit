from airflow._practice import module_getattr
from airflow.providers.standard import operators

__getattr__ = module_getattr("airflow.providers.standard")
