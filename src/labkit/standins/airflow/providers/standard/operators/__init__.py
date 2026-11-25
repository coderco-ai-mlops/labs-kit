from airflow._practice import module_getattr
from airflow.providers.standard.operators import python

__getattr__ = module_getattr("airflow.providers.standard.operators")
