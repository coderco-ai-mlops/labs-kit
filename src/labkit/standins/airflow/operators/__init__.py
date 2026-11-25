from airflow._practice import module_getattr
from airflow.operators import python

__getattr__ = module_getattr("airflow.operators")
